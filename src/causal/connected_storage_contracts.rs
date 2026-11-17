//! Residency, publication, corruption and common-state contracts for connected storage.

use super::connected_checkpoint as checkpoint;
use super::connected_store::Store;
use super::geometry::{Geometry, MATRIX_COUNT, TILES};
use super::scratch_dir::ScratchDir;
use super::store_api::TileStore;
use super::tuning::{self, Knobs, Latent};
use anyhow::Result;
use std::collections::BTreeMap;

#[path = "connected_storage_cache_contracts.rs"]
mod cache;
#[path = "connected_storage_tracking_contracts.rs"]
mod tracking;

/// A unique `work/validation/connected-<name>-…` directory, removed when the guard drops.
pub(super) fn fresh(name: &str) -> Result<ScratchDir> {
    ScratchDir::new(&format!("connected-{name}"))
}

fn common(value: u8) -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        ("common.json".into(), vec![value]),
        ("shared.bin".into(), vec![value; 19]),
    ])
}

fn attach_route_latent(store: &mut Store, value: f32) -> Result<()> {
    let mut lease = store.checkout(3)?;
    let count = lease.tile().geometry.routes(3)? * lease.tile().geometry.width;
    *lease.parts_mut()?.1 = vec![(
        (3, MATRIX_COUNT),
        Latent {
            weights: vec![value; count],
            moments: vec![0.0; count * 2],
            step: 1,
        },
    )];
    store.commit(lease)
}

#[test]
fn leases_bound_residency_and_check_optimizer_only_revisions() -> Result<()> {
    let _guard = tuning::scoped(Knobs::core(0.003, 7));
    let out = fresh("leases")?;
    let mut store = Store::new(&out.join("connected-tiles"), Geometry::fixture(), 7, 4)?;
    let leases = (0..4)
        .map(|id| store.checkout(id))
        .collect::<Result<Vec<_>>>()?;
    assert!(store.checkout(0).is_err());
    assert!(store.checkout(4).is_err());
    assert!(checkpoint::save(&out, &mut store, "identity", 0, &common(0)).is_err());
    assert_eq!(store.stats().peak_live, 4);
    assert_eq!(store.stats().peak_resident, 4);
    for lease in leases {
        store.return_read(lease)?;
    }
    let old_version = store.versions()?[3];
    let old_revision = store.slots[3].meta.revision;
    attach_route_latent(&mut store, 0.1)?;
    assert_eq!(store.versions()?[3], old_version);
    assert_eq!(store.slots[3].meta.revision, old_revision + 1);
    assert_eq!(store.optimizer_steps()?[3], 1);
    let mut lease = store.checkout(3)?;
    assert_eq!(lease.latents()[0].1.weights[0], 0.1);
    lease.parts_mut()?.1[0].1.weights[0] = 0.2;
    assert!(store.return_read(lease).is_err());
    assert_eq!(store.stats().current_live, 0);
    assert!(store.quiescent().is_err());
    Ok(())
}

#[test]
fn checkpoint_restore_is_bounded_and_retention_preserves_runtime_tiles() -> Result<()> {
    let out = fresh("roundtrip")?;
    let geometry = Geometry::fixture();
    let mut store = Store::new(&out.join("connected-tiles"), geometry, 7, 4)?;
    attach_route_latent(&mut store, 0.1)?;
    checkpoint::save(&out, &mut store, "identity", 1, &common(1))?;
    let expected = store
        .slots
        .iter()
        .map(|slot| slot.meta.clone())
        .collect::<Vec<_>>();
    drop(store);
    let loaded = checkpoint::resume(&out, geometry, "identity", 4)?.unwrap();
    assert_eq!(loaded.anchors, 1);
    assert_eq!(loaded.payloads, common(1));
    assert_eq!(loaded.store.stats().peak_live, 1);
    assert_eq!(loaded.store.stats().current_live, 0);
    assert!(loaded.store.stats().peak_resident <= 4);
    assert_eq!(
        loaded
            .store
            .slots
            .iter()
            .map(|slot| slot.meta.clone())
            .collect::<Vec<_>>(),
        expected
    );
    let mut store = loaded.store;
    for anchor in 2..=5 {
        attach_route_latent(&mut store, anchor as f32 * 0.1)?;
        checkpoint::save(&out, &mut store, "identity", anchor, &common(anchor as u8))?;
    }
    assert!(!out.join("checkpoints/a000000000001").exists());
    for id in 0..TILES {
        let lease = store.checkout(id)?;
        store.return_read(lease)?;
    }
    let files =
        std::fs::read_dir(out.join("connected-tiles"))?.collect::<std::io::Result<Vec<_>>>()?;
    assert_eq!(files.len(), TILES);
    let final_state = checkpoint::resume(&out, geometry, "identity", 4)?.unwrap();
    assert_eq!(final_state.anchors, 5);
    assert_eq!(final_state.payloads, common(5));
    assert_eq!(final_state.store.stats().peak_live, 1);
    Ok(())
}

#[test]
fn corrupt_latest_falls_back_without_mixing_and_foreign_identity_is_untouched() -> Result<()> {
    let out = fresh("corruption")?;
    let geometry = Geometry::fixture();
    let mut store = Store::new(&out.join("connected-tiles"), geometry, 7, 4)?;
    checkpoint::save(&out, &mut store, "identity", 1, &common(1))?;
    let before = store.slots[3].meta.clone();
    attach_route_latent(&mut store, 0.25)?;
    let newest = checkpoint::save(&out, &mut store, "identity", 2, &common(2))?;
    std::fs::create_dir(out.join("checkpoints/a000000000003.partial"))?;
    assert!(checkpoint::resume(&out, geometry, "foreign", 4).is_err());
    assert!(newest.exists());
    let manifest = newest.join("manifest.json");
    let current = std::fs::read_to_string(&manifest)?;
    let v4 = current.replace("\"causal-connected-v5\"", "\"causal-connected-v4\"");
    assert_ne!(v4, current);
    std::fs::write(&manifest, v4)?;
    assert!(
        checkpoint::resume(&out, geometry, "identity", 4).is_err(),
        "a v4 checkpoint needs its pinned binary"
    );
    assert!(newest.exists());
    std::fs::write(&manifest, current)?;
    std::fs::write(newest.join("tiles/3.bin"), b"broken")?;
    drop(store);
    let loaded = checkpoint::resume(&out, geometry, "identity", 4)?.unwrap();
    assert_eq!(loaded.anchors, 1);
    assert_eq!(loaded.payloads, common(1));
    assert_eq!(loaded.store.slots[3].meta, before);
    assert!(!newest.exists());
    std::fs::write(loaded.path.join("common.json"), b"damaged")?;
    drop(loaded);
    assert!(checkpoint::resume(&out, geometry, "identity", 4).is_err());
    Ok(())
}

#[test]
fn failed_visit_and_panic_evict_optimizer_and_poison_publication() -> Result<()> {
    let _guard = tuning::scoped(Knobs::core(0.003, 7));
    for panic in [false, true] {
        let out = fresh(if panic { "panic" } else { "error" })?;
        let mut store = Store::new(&out.join("connected-tiles"), Geometry::fixture(), 7, 1)?;
        attach_route_latent(&mut store, 0.1)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.update(3, |_tile| -> Result<()> {
                if panic {
                    panic!("injected connected visit panic");
                }
                anyhow::bail!("injected connected visit failure")
            })
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert!(tuning::steps()
            .iter()
            .all(|((owner, _), _)| *owner >= TILES));
        assert_eq!(store.stats().current_live, 0);
        assert!(store.quiescent().is_err());
    }
    Ok(())
}

#[test]
fn checkpoint_counters_and_report_mechanism_parse_to_the_same_json() -> Result<()> {
    // scripts/core_owner_checkpoint.py admits recovery only if connected.json `counters` and
    // report.json `mechanism` parse equal; the two writers must print every float the same way.
    let counters = super::connected_state::Counters {
        master_rms_max: [
            0.8164966106414795,
            1.4999998807907104,
            0.1 + 0.2,
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        readout_rms_sum: 1234.5678901234,
        local_applied_norm_sum: 23851.351831081054,
        local_updates: 7,
        ..Default::default()
    };
    let direct: serde_json::Value = serde_json::from_slice(&serde_json::to_vec(&counters)?)?;
    let report = serde_json::json!({ "mechanism": counters });
    let via_value: serde_json::Value = serde_json::from_str(&report["mechanism"].to_string())?;
    assert_eq!(direct, via_value);
    Ok(())
}
