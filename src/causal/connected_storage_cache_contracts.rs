//! Cache parity, bounded eviction and failed-write contracts.

use super::*;

fn change(store: &mut Store, id: usize, step: u64) -> Result<()> {
    let mut lease = store.checkout(id)?;
    let (tile, latents) = lease.parts_mut()?;
    let matrix = &tile.matrices[0];
    let mut values = (0..matrix.len())
        .map(|index| matrix.value(index))
        .collect::<Result<Vec<_>>>()?;
    values[0] = (step % 3) as i8 - 1;
    tile.matrices[0] =
        crate::ternary::TernaryMatrix::from_values(matrix.rows(), matrix.cols(), &values)?;
    tile.version += 1;
    let count = tile.geometry.routes(id)? * tile.geometry.width;
    *latents = vec![(
        (id, MATRIX_COUNT),
        Latent {
            weights: vec![step as f32 * 0.01; count],
            moments: vec![step as f32 * 0.001; count * 2],
            step,
        },
    )];
    store.commit(lease)
}

#[test]
fn cached_updates_match_eager_payloads_with_less_io() -> Result<()> {
    let out = fresh("cache-parity")?;
    let geometry = Geometry::fixture();
    let mut cached = Store::new(&out.join("cached"), geometry, 7, 4)?;
    let mut eager = Store::new(&out.join("eager"), geometry, 7, 4)?;
    eager.set_eager(true)?;
    for (index, id) in [3, 3, 3, 0, 1, 2, 4, 3, 6, 5, 3].into_iter().enumerate() {
        change(&mut cached, id, index as u64 + 1)?;
        change(&mut eager, id, index as u64 + 1)?;
        assert_eq!(cached.versions()?, eager.versions()?);
        assert_eq!(cached.optimizer_steps()?, eager.optimizer_steps()?);
        assert!(cached.stats().current_resident <= 4);
    }
    cached.flush()?;
    let metadata = |store: &Store| {
        store
            .slots
            .iter()
            .map(|slot| slot.meta.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(metadata(&cached), metadata(&eager));
    assert_eq!(
        cached.stats().trit_transitions,
        eager.stats().trit_transitions
    );
    assert_eq!(cached.stats().trit_reversals, eager.stats().trit_reversals);
    assert_eq!(cached.stats().peak_resident, 4);
    assert!(cached.stats().evictions > 0 && cached.stats().cache_hits > 0);
    assert!(cached.stats().reads < eager.stats().reads);
    assert!(cached.stats().writes < eager.stats().writes);
    let reads = cached.stats().reads;
    let writes = cached.stats().writes;
    for _ in 0..4 {
        let lease = cached.checkout(3)?;
        cached.return_read(lease)?;
    }
    assert_eq!(
        (cached.stats().reads, cached.stats().writes),
        (reads, writes)
    );
    Ok(())
}

#[test]
fn dirty_eviction_and_checkpoint_flush_fail_closed() -> Result<()> {
    for checkpointing in [false, true] {
        let out = fresh(if checkpointing {
            "flush-error"
        } else {
            "eviction-error"
        })?;
        let root = out.join("connected-tiles");
        let mut store = Store::new(&root, Geometry::fixture(), 7, 1)?;
        let old = store.slots[3].clone();
        change(&mut store, 3, 1)?;
        assert!(store.slots[3].meta.hash.is_empty());
        let reads = store.stats().reads;
        std::fs::create_dir(root.join("tile-3-1.partial"))?;
        if checkpointing {
            assert!(checkpoint::save(&out, &mut store, "identity", 1, &common(1)).is_err());
        } else {
            assert!(store.checkout(0).is_err());
        }
        assert!(!out.join("checkpoints/a000000000001").exists());
        assert_eq!(store.stats().reads, reads);
        assert_eq!(store.stats().current_live, 0);
        assert_eq!(store.stats().current_resident, 1);
        assert_eq!(store.stats().peak_resident, 1);
        assert_eq!(
            super::super::connected_checkpoint_io::hash(&old.path, 1 << 20)?,
            old.meta.hash
        );
        assert!(store.quiescent().is_err());
        assert!(store.checkout(3).is_err());
    }
    Ok(())
}

#[test]
fn immutable_read_rejects_and_cleans_hidden_optimizer_mutation() -> Result<()> {
    let _guard = tuning::scoped(Knobs::core(0.003, 7));
    for outcome in 0..3 {
        let out = fresh("read-optimizer-mutation")?;
        let mut store = Store::new(&out.join("connected-tiles"), Geometry::fixture(), 7, 1)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.read(3, |_tile| -> Result<()> {
                tuning::put_tile(
                    3,
                    vec![(
                        (3, 0),
                        Latent {
                            weights: vec![0.0],
                            moments: vec![0.0; 2],
                            step: 1,
                        },
                    )],
                )?;
                match outcome {
                    0 => Ok(()),
                    1 => anyhow::bail!("injected read error"),
                    _ => panic!("injected read panic"),
                }
            })
        }));
        if outcome == 2 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert!(tuning::take_tile(3).is_empty());
        assert_eq!(store.stats().current_live, 0);
        assert!(store.quiescent().is_err());
    }
    Ok(())
}
