use super::math::fingerprint;
use super::{Profile, CONTEXT};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub(super) struct Corpus {
    pub train: Vec<u8>,
    pub valid: Vec<u8>,
}

impl Corpus {
    pub fn validate_profile(&self, profile: Profile) -> Result<()> {
        if profile == Profile::T2Credit {
            ensure!(
                self.train.as_slice() == include_bytes!("../../datasets/tiny/train.txt"),
                "T2 training data mismatch"
            );
            ensure!(
                self.valid.as_slice() == include_bytes!("../../datasets/tiny/development-t2.txt"),
                "T2 development data mismatch"
            );
        }
        Ok(())
    }

    pub fn load(train: &Path, valid: &Path) -> Result<Self> {
        ensure!(
            train.canonicalize()? != valid.canonicalize()?,
            "training and validation must differ"
        );
        let train = read(train)?;
        let valid = read(valid)?;
        ensure!(
            train != valid,
            "training and validation contents must differ"
        );
        Ok(Self { train, valid })
    }

    pub fn identity(&self) -> Value {
        let evaluated_bytes = CONTEXT + (self.valid.len() - CONTEXT).min(64);
        json!({
            "unit": "raw UTF-8 byte", "vocabulary": 256,
            "train_bytes": self.train.len(), "valid_bytes": self.valid.len(),
            "identity_algorithm": "fnv1a64-noncryptographic",
            "train_identity": format!("{:016x}", fingerprint(&self.train)),
            "valid_identity": format!("{:016x}", fingerprint(&self.valid)),
            "evaluation_examples": (self.valid.len() - CONTEXT).min(64),
            "evaluation_prefix": {"offset": 0, "byte_count": evaluated_bytes,
                "identity_fnv1a64": format!("{:016x}", fingerprint(&self.valid[..evaluated_bytes]))}
        })
    }
}

fn read(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(65_537).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 65_536, "text data exceeds 64 KiB");
    ensure!(bytes.len() > CONTEXT, "text data needs at least nine bytes");
    std::str::from_utf8(&bytes).context("text data must be UTF-8")?;
    Ok(bytes)
}

pub(super) fn example(bytes: &[u8], index: usize) -> (&[u8], usize) {
    let start = index % (bytes.len() - CONTEXT);
    (
        &bytes[start..start + CONTEXT],
        usize::from(bytes[start + CONTEXT]),
    )
}

pub(super) fn context(bytes: &[u8], index: usize) -> &[u8] {
    let start = index % (bytes.len() - CONTEXT);
    &bytes[start..start + CONTEXT]
}
