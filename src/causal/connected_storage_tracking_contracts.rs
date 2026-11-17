//! Persisted trit reversals and optimizer corruption rejection.

use super::*;

fn set_first(store: &mut Store, value: i8) -> Result<()> {
    let mut lease = store.checkout(3)?;
    let tile = lease.parts_mut()?.0;
    let matrix = &tile.matrices[0];
    let mut values = (0..matrix.len())
        .map(|index| matrix.value(index))
        .collect::<Result<Vec<_>>>()?;
    values[0] = value;
    tile.matrices[0] =
        crate::ternary::TernaryMatrix::from_values(matrix.rows(), matrix.cols(), &values)?;
    tile.version += 1;
    store.commit(lease)
}

#[test]
fn trit_reversal_remembers_the_pre_checkpoint_direction() -> Result<()> {
    let out = fresh("reversal")?;
    let geometry = Geometry::fixture();
    let mut store = Store::new(&out.join("connected-tiles"), geometry, 7, 4)?;
    let lease = store.checkout(3)?;
    let original = lease.tile().matrices[0].value(0)?;
    store.return_read(lease)?;
    let next = if original == 1 { 0 } else { original + 1 };
    set_first(&mut store, next)?;
    assert_eq!(store.stats().trit_transitions, 1);
    assert_eq!(store.stats().trit_reversals, 0);
    checkpoint::save(&out, &mut store, "identity", 1, &common(1))?;
    drop(store);
    let mut store = checkpoint::resume(&out, geometry, "identity", 4)?
        .unwrap()
        .store;
    assert_eq!(store.stats().trit_transitions, 0);
    set_first(&mut store, original)?;
    assert_eq!(store.stats().trit_transitions, 1);
    assert_eq!(store.stats().trit_reversals, 1);
    Ok(())
}

#[test]
fn negative_second_moment_is_rejected_by_both_payload_boundaries() -> Result<()> {
    let geometry = Geometry::fixture();
    for writing in [false, true] {
        let out = fresh(if writing {
            "negative-write"
        } else {
            "negative-read"
        })?;
        let mut store = Store::new(&out.join("connected-tiles"), geometry, 7, 4)?;
        store.set_eager(true)?;
        attach_route_latent(&mut store, 0.1)?;
        if writing {
            let mut lease = store.checkout(3)?;
            lease.parts_mut()?.1[0].1.moments[1] = -1.0;
            assert!(store.commit(lease).is_err());
        } else {
            let slot = &mut store.slots[3];
            let mut bytes = std::fs::read(&slot.path)?;
            let packed = super::super::connected_store_codec::packed_length(geometry, 3)?;
            let length = geometry.routes(3)? * geometry.width;
            let offset = 16 + packed + 8 + 32 + length * 4 + 4;
            bytes[offset..offset + 4].copy_from_slice(&(-1.0_f32).to_le_bytes());
            std::fs::write(&slot.path, &bytes)?;
            slot.meta.hash = super::super::storage_paths::digest(&bytes);
            assert!(store.checkout(3).is_err());
        }
        assert!(store.quiescent().is_err());
    }
    Ok(())
}
