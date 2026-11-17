//! Test-only paging snapshot and panic cleanup contract; see AGENTS.md §Core v2.

use super::super::scratch_dir::ScratchDir;
use super::*;
use sha2::{Digest, Sha256};

pub(super) fn latent_digest(values: &[((usize, usize), tuning::Latent)]) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update((values.len() as u64).to_le_bytes());
    for ((owner, matrix), latent) in values {
        ensure!(
            latent.moments.len() == 2 * latent.weights.len(),
            "latent moment shape"
        );
        for value in [
            *owner as u64,
            *matrix as u64,
            latent.step,
            latent.weights.len() as u64,
        ] {
            hash.update(value.to_le_bytes());
        }
        for value in latent.weights.iter().chain(&latent.moments) {
            hash.update(value.to_le_bytes());
        }
    }
    Ok(hash.finalize().into())
}

pub(super) fn fingerprint_hashes(
    state: &super::super::state::State,
    store: &super::super::resident::ResidentStore,
) -> Result<Vec<String>> {
    Ok(super::super::core_contracts::fingerprint(state, store)?
        .into_iter()
        .map(|bytes| digest(&bytes))
        .collect())
}

impl Paged {
    pub(super) fn finish_visit<T>(
        &mut self,
        id: usize,
        visit: impl FnOnce() -> Result<T>,
    ) -> (Result<T>, Latents) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(visit));
        let values = self.leave(id);
        match result {
            Ok(result) => (result, values),
            Err(error) => {
                self.poisoned = true;
                std::panic::resume_unwind(error)
            }
        }
    }

    pub(super) fn snapshot(&self) -> Result<(super::super::resident::ResidentStore, Latents)> {
        let mut tiles = Vec::new();
        let mut latents = Vec::new();
        for id in 0..TILES {
            let (tile, mut values) = self.load(id)?;
            tiles.push(tile);
            latents.append(&mut values);
        }
        Ok((
            super::super::resident::ResidentStore::from_tiles(tiles, self.geometry)?,
            latents,
        ))
    }
}

#[test]
fn read_detects_optimizer_mutation_without_retaining_serialized_state() -> Result<()> {
    use super::super::core_codec::encode_latents;
    let geometry = Geometry::fixture();
    let _guard = tuning::scoped(super::super::tuning::Knobs::core(0.003, 7));
    let scratch = ScratchDir::new("core-paged-read-mutation")?;
    let mut store = Paged::new(&scratch.path().join("store"), geometry, 7)?;
    store.update(3, |tile| {
        tile.apply(&super::super::tile_gradients::TileGradients::zeros(
            geometry, 3,
        )?)?;
        Ok(())
    })?;
    let (_, values) = store.load(3)?;
    assert!(!values.is_empty());
    let expected: [u8; 32] = Sha256::digest(encode_latents(&values)?).into();
    assert_eq!(latent_digest(&values)?, expected);
    drop(values);
    let before = fs::read(&store.slots[3].path)?;
    let result = store.read(3, |_| {
        let mut values = tuning::take_tile(3);
        values[0].1.weights[0] += 1.0;
        tuning::put_tile(3, values)?;
        Ok(())
    });
    assert!(result.is_err());
    assert!(!store.live && store.versions().is_err());
    assert_eq!(fs::read(&store.slots[3].path)?, before);
    assert!(tuning::steps()
        .iter()
        .all(|((owner, _), _)| *owner >= TILES));
    Ok(())
}

#[test]
fn callback_failures_detach_latents_and_poison_store() -> Result<()> {
    let geometry = Geometry::fixture();
    let _guard = tuning::scoped(super::super::tuning::Knobs::core(0.003, 7));
    for panic in [false, true] {
        let scratch = ScratchDir::new(&format!("core-paged-failure-{panic}"))?;
        let mut store = Paged::new(&scratch.path().join("store"), geometry, 7)?;
        store.update(3, |tile| {
            tile.apply(&super::super::tile_gradients::TileGradients::zeros(
                geometry, 3,
            )?)?;
            Ok(())
        })?;
        let original = fs::read(&store.slots[0].path)?;
        fs::write(&store.slots[0].path, vec![0; 256 * 1024])?;
        assert!(store.read(0, |_| Ok(())).is_err());
        fs::write(&store.slots[0].path, original)?;
        let (_, values) = store.load(3)?;
        tuning::put_tile(3, values)?;
        let before_steps = tuning::steps();
        assert!(tuning::put_tile(3, Vec::new()).is_err());
        assert!(store.read(3, |_| Ok(())).is_err());
        assert_eq!(tuning::steps(), before_steps);
        drop(tuning::take_tile(3));
        let before = fs::read(&store.slots[3].path)?;
        if panic {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _: Result<()> = store.read(3, |_| panic!("injected tile panic"));
            }));
            assert!(result.is_err());
        } else {
            let result: Result<()> = store.update(3, |tile| {
                tile.apply(&super::super::tile_gradients::TileGradients::zeros(
                    geometry, 3,
                )?)?;
                anyhow::bail!("injected failure after mutation")
            });
            assert!(result.is_err());
        }
        assert!(!store.live && store.versions().is_err());
        assert_eq!(fs::read(&store.slots[3].path)?, before);
        assert!(tuning::steps()
            .iter()
            .all(|((owner, _), _)| *owner >= TILES));
    }
    Ok(())
}

#[test]
fn failed_publication_keeps_old_slot_and_detaches_latents() -> Result<()> {
    let geometry = Geometry::fixture();
    let _guard = tuning::scoped(super::super::tuning::Knobs::core(0.003, 7));
    let scratch = ScratchDir::new("core-paged-publish-failure")?;
    let directory = scratch.path().join("store");
    let mut store = Paged::new(&directory, geometry, 7)?;
    let gradients = super::super::tile_gradients::TileGradients::zeros(geometry, 3)?;
    store.update(3, |tile| tile.apply(&gradients))?;
    let old_path = store.slots[3].path.clone();
    let old_hash = store.slots[3].hash.clone();
    let old_version = store.slots[3].version;
    let old_steps = store.slots[3].steps;
    let old_bytes = fs::read(&old_path)?;
    let conflict = directory.join(format!("tile-3-{}.bin", store.generation + 1));
    fs::create_dir(&conflict)?;

    assert!(store.update(3, |tile| tile.apply(&gradients)).is_err());
    assert!(store.poisoned && !store.live && store.versions().is_err());
    assert_eq!(store.slots[3].path, old_path);
    assert_eq!(store.slots[3].hash, old_hash);
    assert_eq!(store.slots[3].version, old_version);
    assert_eq!(store.slots[3].steps, old_steps);
    assert_eq!(fs::read(&old_path)?, old_bytes);
    assert!(tuning::steps()
        .iter()
        .all(|((owner, _), _)| *owner >= TILES));
    Ok(())
}
