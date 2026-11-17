//! Bounded fingerprints of actual cached tile and optimizer state; see CONNECTED.md.

use super::{codec, Store};
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};

impl Store {
    pub fn fingerprint(&mut self) -> Result<String> {
        self.quiescent()?;
        ensure!(
            self.slots.iter().all(|slot| !slot.meta.hash.is_empty()),
            "fingerprint requires durably published tile state"
        );
        let mut digest = Sha256::new();
        for id in 0..self.slots.len() {
            let metadata = serde_json::to_vec(&self.slots[id].meta)?;
            let lease = self.checkout(id)?;
            let result = codec::digest(&lease.tile, &lease.latents, &lease.last_direction);
            let hash = match result {
                Ok(hash) if hash == self.slots[id].meta.hash => hash,
                Ok(_) => {
                    self.abort(id);
                    anyhow::bail!("cached connected payload differs from its published hash");
                }
                Err(error) => {
                    self.abort(id);
                    return Err(error);
                }
            };
            digest.update((id as u64).to_le_bytes());
            digest.update(metadata);
            digest.update(hash.as_bytes());
            self.return_read(lease)?;
        }
        self.quiescent()?;
        Ok(format!("{:x}", digest.finalize()))
    }
}
