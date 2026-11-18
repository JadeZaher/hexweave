//! Golden digests of connected batch-1 training trajectories; moving one means numerics changed.

use super::connected_bias_contracts::corpus;
use super::connected_config::Options;
use super::connected_state::State;
use super::connected_step;
use super::connected_storage_contracts::fresh;
use super::connected_store::Store;
use super::geometry::Geometry;
use super::storage_paths::digest;
use super::tuning::{self, Knobs};
use anyhow::Result;
use serde_json::json;

/// FP32 `anchor_exact` with discrete restarts, restart replay and prioritized replay.
const GOLDEN_EXACT: &str = "285e4495cc38b7f2f5fc0cb59552019550b533fd4c5a5a7cefbc7f6f3acd8cfe";

/// Default ternary `lanes` credit.
const GOLDEN_LANES: &str = "0c4c2c1fcad73206e69620b1d1925389b9f8b5b6a0003d6b10b03287b3aae631";

/// SHA-256 over per-anchor loss bits, every state payload and every flushed tile's metadata.
fn trajectory(name: &str, options: serde_json::Value, anchors: usize) -> Result<String> {
    let options: Options = serde_json::from_value(options)?;
    let fp32 = options.precision == super::connected_precision::Precision::Fp32;
    let _guard = tuning::scoped(Knobs {
        connected: true,
        connected_fp32: fp32,
        ..Knobs::core(0.003, 7)
    });
    let out = fresh(name)?;
    let text: Vec<u8> = b"the cat sat on the mat. "
        .iter()
        .cycle()
        .take(512)
        .copied()
        .collect();
    let corpus = corpus(&out.join("corpus"), &text)?;
    let geometry = Geometry::fixture();
    let mut state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let root = out.join("connected-tiles");
    let mut store = Store::new_precision(&root, geometry, 7, options.workers, options.precision)?;
    store.set_paging(options.resident(), options.write_behind_tiles)?;
    let mut bytes = Vec::new();
    for _ in 0..anchors {
        let loss = connected_step::step(&mut state, &mut store, &corpus, options, 0.003, false)?;
        bytes.extend(loss.to_bits().to_le_bytes());
    }
    // The restart case must actually restart, so the digest covers `restart::finish`.
    let restarted = state
        .restarts
        .iter()
        .flatten()
        .any(|entry| entry.restarts > 0);
    assert_eq!(restarted, options.restart.is_some(), "{name}");
    for (name, payload) in state.payloads()? {
        bytes.extend(name.as_bytes());
        bytes.extend(payload);
    }
    store.flush()?;
    for slot in &store.slots {
        bytes.extend(serde_json::to_vec(&slot.meta)?);
    }
    Ok(digest(&bytes))
}

#[test]
fn connected_batch_one_trajectories_match_their_golden_digests() -> Result<()> {
    let exact = trajectory(
        "golden-exact",
        json!({"precision":"fp32","lanes":2,"workers":2,"visits":4,"credit_mode":"anchor_exact",
            "replay":{"capacity":16,"per_anchor":2},
            "restart":{"floor":0.1,"threshold":0.01,"adapt":0.25,"replay":1}}),
        16,
    )?;
    let lanes = trajectory("golden-lanes", json!({"lanes":2,"workers":2,"visits":2}), 8)?;
    // The goldens are seven-tile numerics; update one only when the change is intended.
    if !cfg!(feature = "tiles19") {
        assert_eq!(
            (exact.as_str(), lanes.as_str()),
            (GOLDEN_EXACT, GOLDEN_LANES),
            "connected batch-1 numerics changed; this also moves with the platform's libm"
        );
    }
    Ok(())
}
