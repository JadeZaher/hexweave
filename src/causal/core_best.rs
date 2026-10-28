//! Best-checkpoint selection for science runs: the lowest dev NLL is kept under `out/best` with an
//! identity-stamped `selection.json`; see AGENTS.md §core.

use super::core_checkpoint;
use super::resident::ResidentStore;
use super::state::State;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::path::Path;

/// Lowest dev NLL this run identity recorded; infinity if none, unreadable or another run's.
pub(super) fn best_so_far(out: &Path, identity: &str) -> f64 {
    std::fs::read(out.join("best").join("selection.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .filter(|selection| selection["identity_sha256"] == identity)
        .and_then(|selection| selection["dev_nll_per_byte"].as_f64())
        .unwrap_or(f64::INFINITY)
}

/// A fresh run must not inherit another run's best checkpoints.
pub(super) fn ensure_fresh(out: &Path) -> Result<()> {
    let best = out.join("best");
    if best.exists() && std::fs::read_dir(&best)?.next().is_some() {
        bail!(
            "{} holds best checkpoints; set resume or choose a new out",
            best.display()
        );
    }
    Ok(())
}

/// Keep a better science state; a failure is logged and never stops training.
pub(super) fn keep(
    out: &Path,
    state: &State,
    store: &ResidentStore,
    identity: &str,
    line: &Value,
    best: &mut f64,
) {
    if let Err(error) = try_keep(out, state, store, identity, line, best) {
        eprintln!("best checkpoint skipped: {error:#}");
    }
}

fn try_keep(
    out: &Path,
    state: &State,
    store: &ResidentStore,
    identity: &str,
    line: &Value,
    best: &mut f64,
) -> Result<()> {
    let Some(score) = line["eval"]["dev_nll_per_byte"].as_f64() else {
        return Ok(());
    };
    let anchors = state.global_opportunity;
    if anchors == 0 || !score.is_finite() || score >= *best {
        return Ok(());
    }
    let root = out.join("best");
    if !root
        .join("checkpoints")
        .join(format!("a{anchors:012}"))
        .exists()
    {
        core_checkpoint::save(&root, state, store, identity)?;
    }
    let selection =
        json!({ "anchors": anchors, "dev_nll_per_byte": score, "identity_sha256": identity });
    core_checkpoint::publish(
        &root.join("selection.json"),
        &serde_json::to_vec_pretty(&selection)?,
    )?;
    *best = score;
    Ok(())
}
