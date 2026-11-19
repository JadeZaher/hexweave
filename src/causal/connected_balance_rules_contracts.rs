//! Pure balance rules: eviction order and the multiplier/bonus clamps and neutral cases.

use super::super::connected_signal::Kind;
use super::super::geometry::START_TILE;
use super::*;
use serde_json::json;

#[test]
fn balanced_eviction_keeps_a_rare_tiles_record_that_fifo_drops() -> Result<()> {
    let geometry = Geometry::fixture();
    let width = geometry.width;
    let record = |tile: usize, epoch: u64| Record {
        input: vec![0.0; width],
        output: vec![0.0; width],
        history: vec![0.0; width],
        correction: vec![0.0; width],
        tile_id: tile,
        revision: 0,
        shared_revision: epoch,
        epoch,
        kind: Kind::Observed,
        lane: None,
    };
    for balanced in [false, true] {
        let mut memory = Memory::new(geometry, 8)?;
        memory.balanced = balanced;
        memory.publish(record(5, 0))?;
        // Two busy tiles publish more than the capacity, inside the 64-epoch age window.
        for epoch in 1..=12 {
            let before: Vec<u64> = memory.records.iter().map(|r| r.epoch).collect();
            memory.publish(record(epoch as usize % 2, epoch))?;
            if epoch == 8 {
                // The first eviction: FIFO drops the rare epoch 0; balanced drops busy tile 1's
                // oldest (4 records against tile 0's 3), epoch 1.
                let kept: Vec<u64> = memory.records.iter().map(|r| r.epoch).collect();
                let gone: Vec<u64> = before.into_iter().filter(|e| !kept.contains(e)).collect();
                assert_eq!(gone, vec![u64::from(balanced)]);
            }
        }
        memory.validate()?;
        let rare = memory.records.iter().filter(|r| r.tile_id == 5).count();
        assert_eq!((memory.records.len(), rare), (8, usize::from(balanced)));
        assert_eq!(memory.publications, 13);
    }
    Ok(())
}

#[test]
fn balance_multiplier_and_bonus_hit_their_clamps_and_stay_neutral_where_defined() -> Result<()> {
    let options: Options = serde_json::from_value(json!({"credit_mode":"anchor_exact",
        "balance":{"horizon":64,"rate":1.0,"route":2.0}}))?;
    let outer = (0..TILES)
        .find(|&tile| tile != START_TILE)
        .expect("an outer tile");
    let estimate = |samples: u64, shares: [f64; 2]| {
        let mut estimate = Estimate {
            samples,
            ..Estimate::default()
        };
        estimate.share = [shares[1]; TILES];
        estimate.share[outer] = shares[0];
        estimate.share[START_TILE] = 1.0;
        estimate
    };
    // ((samples, [tested outer tile share, other shares]), its multiplier, its bonus)
    for ((samples, shares), factor, steer) in [
        // A never-visited tile: ratio far above 4, both upper clamps.
        ((64, [0.0, 0.5]), 4.0, 2.0),
        // No outer visits at all: the mean is below the floor, so everything is neutral.
        ((64, [0.0, 0.0]), 1.0, 0.0),
        // One very busy tile: ratio far below 1/4, both lower clamps.
        ((64, [1.0, 0.01]), 0.25, -2.0),
        // Uniform outer shares: ratio exactly 1.
        ((64, [0.5, 0.5]), 1.0, 0.0),
        // Before `horizon` anchors the estimate is not used.
        ((63, [0.0, 0.5]), 1.0, 0.0),
    ] {
        let estimate = estimate(samples, shares);
        let bonuses = bonus(&options, Some(&estimate)).expect("route bonus");
        let shown = format!("{samples} {shares:?}");
        assert_eq!(
            multiplier(&options, Some(&estimate), outer)?,
            factor,
            "{shown}"
        );
        assert_eq!(bonuses[outer], steer, "{shown}");
        assert_eq!(
            multiplier(&options, Some(&estimate), START_TILE)?,
            1.0,
            "{shown}"
        );
        assert_eq!(bonuses[START_TILE], 0.0, "{shown}");
    }
    Ok(())
}
