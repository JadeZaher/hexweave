//! Best-checkpoint selection (science and corpus runs): the lowest `dev_nll_per_byte` is kept under `out/best`; see AGENTS.md §core.

use super::core_checkpoint;
use super::resident::ResidentStore;
use super::state::State;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::path::Path;

/// Bump when evaluation numerics change (2 = history replay) so an earlier eval version's selection is never compared against the new NLL scale.
const EVAL_VERSION: u64 = 2;

/// A stored selection's held-out sample size; a missing field predates `eval_spans` and counts as 16.
fn stored_eval_spans(selection: &Value) -> u64 {
    selection["eval_spans"].as_u64().unwrap_or(16)
}

/// Lowest dev NLL this run identity recorded at the current eval version and `eval_spans`
/// (`Config::eval_spans`, so a differently-sized held-out sample is never compared); infinity if
/// none, unreadable, foreign or stale.
pub(super) fn best_so_far(out: &Path, identity: &str, eval_spans: usize) -> f64 {
    std::fs::read(out.join("best").join("selection.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .filter(|selection| {
            selection["identity_sha256"] == identity
                && selection["eval_version"] == EVAL_VERSION
                && stored_eval_spans(selection) == eval_spans as u64
        })
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
    eval_spans: usize,
) {
    if let Err(error) = try_keep(out, state, store, identity, line, best, eval_spans) {
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
    eval_spans: usize,
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
    let selection = json!({ "anchors": anchors, "dev_nll_per_byte": score,
        "identity_sha256": identity, "eval_version": EVAL_VERSION, "eval_spans": eval_spans });
    core_checkpoint::publish(
        &root.join("selection.json"),
        &serde_json::to_vec_pretty(&selection)?,
    )?;
    *best = score;
    Ok(())
}
