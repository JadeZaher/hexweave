//! Frozen public held-out and past-training retention assessment; see CONNECTED.md.

use super::connected_eval_config::Options;
use super::connected_state::State;
use super::connected_store::Store;
use super::core_corpus::{Corpus, SPAN_BYTES};
use super::decoding::{story_opening, Decoder};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(test)]
#[path = "connected_eval_contracts.rs"]
mod contracts;
#[path = "connected_eval_probe.rs"]
mod probe;
#[path = "connected_eval_replay.rs"]
mod replay;

const PROMPTS: [&str; 4] = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "Tom and his dog went to the",
    "The sun was",
];

struct Budget<'a> {
    started: Instant,
    until: Duration,
    out: Option<&'a Path>,
}

impl Budget<'_> {
    fn interrupted(&self) -> bool {
        self.started.elapsed() >= self.until
            || self.out.is_some_and(|out| out.join("STOP").exists())
    }
}

pub(super) fn retention_due(anchors: u64, options: Options) -> bool {
    probe::RETENTION_OFFSETS
        .iter()
        .take(options.retention_spans)
        .any(|offset| offset + SPAN_BYTES as u64 == anchors)
}

/// Evaluation budgets and probe offsets are reported independently of the learning identity.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate(
    state: &State,
    store: &mut Store,
    corpus: &Corpus,
    spans: usize,
    out: &Path,
    cases: usize,
    options: Options,
    retention_only: bool,
) -> Result<Value> {
    options.validate()?;
    ensure!(
        (1..=128).contains(&spans),
        "connected held-out spans must be in 1..=128"
    );
    ensure!(
        (1..=PROMPTS.len()).contains(&cases),
        "connected sample cases must be in 1..=4"
    );
    let started = Instant::now();
    state.validate()?;
    store.flush()?;
    store.quiescent()?;
    let before = state.payloads()?;
    let tiles: Vec<_> = store.slots.iter().map(|slot| slot.meta.clone()).collect();
    let whole = Budget {
        started,
        until: Duration::from_secs(options.budget_seconds),
        out: Some(out),
    };
    let result = measure(
        state,
        store,
        corpus,
        spans,
        cases,
        options,
        retention_only,
        &whole,
    )?;
    store.quiescent()?;
    ensure!(
        before == state.payloads()?,
        "evaluation mutated connected training state"
    );
    ensure!(
        tiles
            == store
                .slots
                .iter()
                .map(|slot| slot.meta.clone())
                .collect::<Vec<_>>(),
        "evaluation changed connected tiles"
    );
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn measure(
    state: &State,
    store: &mut Store,
    corpus: &Corpus,
    spans: usize,
    cases: usize,
    options: Options,
    retention_only: bool,
    whole: &Budget<'_>,
) -> Result<Value> {
    let retained = probe::retention(
        corpus,
        options.retention_spans,
        state.anchors,
        retention_only,
    )?;
    let budget = |fraction: f64| Budget {
        started: whole.started,
        until: Duration::from_secs_f64(options.budget_seconds as f64 * fraction),
        out: whole.out,
    };
    let heldout = if retention_only {
        Vec::new()
    } else {
        probe::heldout(corpus, spans)
    };
    let fractions = if retention_only {
        [0.0, 0.25, 0.0]
    } else if options.retention_spans == 0 {
        [0.75, 0.0, 0.25]
    } else {
        [0.60, 0.25, 0.15]
    };
    let dev = probe::score(state, store, &heldout, corpus.window, &budget(fractions[0]))?;
    // The read-free score gets whatever the same cumulative deadline leaves; see its `partial`.
    let shared = if state.sections.is_some() && !retention_only {
        let window = corpus.window;
        let without =
            probe::score_with(state, store, &heldout, window, &budget(fractions[0]), false)?;
        Some(json!({"heldout_without_shared": without["bits_per_byte"],
            "heldout_without_shared_partial": without["partial"],
            "tile_similarity": probe::similarity(state, store, &heldout, window)?,
            "tile_similarity_positions": probe::SIMILARITY_POSITIONS}))
    } else {
        None
    };
    let retention_fraction = fractions[0] + fractions[1];
    let retention = probe::score(
        state,
        store,
        &retained,
        corpus.window,
        &budget(retention_fraction),
    )?;
    let samples = if retention_only {
        Vec::new()
    } else {
        samples(
            state,
            store,
            &corpus.heldout,
            corpus.window,
            cases,
            options.sample_bytes,
            &budget(1.0),
        )?
    };
    let requested_samples = if retention_only { 0 } else { cases };
    let mut result = json!({
        "schema": 2, "kind": if retention_only { "first_exposure_retention" } else { "full" },
        "dev_nll_per_byte": dev["nll_per_byte"], "dev_bits_per_byte": dev["bits_per_byte"],
        "heldout_targets": dev["scored_targets"], "heldout": dev, "retention": retention,
        "retention_requested_offsets": &probe::RETENTION_OFFSETS[..options.retention_spans],
        "retention_eligible_spans": retained.len(), "retention_reference": "fixed-already-observed-first-pass-spans",
        "samples": samples, "requested_samples": requested_samples,
        "partial": dev["partial"].as_bool().unwrap_or(true) || retention["partial"].as_bool().unwrap_or(true)
            || samples.len() < requested_samples || samples.iter().any(|sample| sample["stop_reason"] == "evaluation_budget_or_stop"),
        "measurement_options": options, "heldout_requested_spans": heldout.len(), "span_bytes": SPAN_BYTES,
        "elapsed_seconds": whole.started.elapsed().as_secs_f64(),
        "budget_fractions": fractions,
        "memory_policy": "isolated-observe-then-frozen-local-lanes-v1",
        "forward": store.precision, "weights_and_predictor_frozen": true,
    });
    if let Some(shared) = shared {
        result["shared"] = shared;
    }
    Ok(result)
}

/// Nucleus-sampled story continuations after a full held-out window (AGENTS.md §Generation).
#[allow(clippy::too_many_arguments)]
fn samples(
    state: &State,
    store: &mut Store,
    heldout: &[u8],
    window: usize,
    cases: usize,
    limit: usize,
    budget: &Budget<'_>,
) -> Result<Vec<Value>> {
    let mut samples = Vec::new();
    for (slot, prompt) in PROMPTS.iter().take(cases).enumerate() {
        if budget.interrupted() {
            break;
        }
        let opening = story_opening(heldout, window, slot, cases, prompt.as_bytes())?;
        let context = &opening[..opening.len() - prompt.len()];
        let decoder = Decoder::Nucleus {
            temperature: 0.8,
            top_p: 0.9,
            seed: slot as u64,
        };
        let decode = super::connected_decode::Options {
            decoder,
            ..Default::default()
        };
        let mut replay = replay::Replay::new(state, store)?;
        samples.push(replay.generate(context, prompt, window, limit, budget, decode)?);
    }
    Ok(samples)
}

/// Test probe: frozen-replay history inputs at `bytes` after scoring `observed` pairs.
#[cfg(test)]
pub(super) fn frozen_histories(
    state: &State,
    store: &mut Store,
    observed: &[(&[u8], u8)],
    bytes: &[u8],
) -> Result<Vec<Vec<f32>>> {
    replay::Replay::new(state, store)?.histories(observed, bytes)
}

/// Frozen-weight generation of `prompt` after `context` within a deadline.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_frozen_with_options(
    state: &State,
    store: &mut Store,
    context: &[u8],
    prompt: &str,
    window: usize,
    limit: usize,
    started: Instant,
    timeout: Duration,
    decode: super::connected_decode::Options,
) -> Result<Value> {
    let budget = Budget {
        started,
        until: timeout,
        out: None,
    };
    let mut replay = replay::Replay::new(state, store)?;
    let mut result = replay.generate(context, prompt, window, limit, &budget, decode)?;
    if result["stop_reason"] == "evaluation_budget_or_stop" {
        result["stop_reason"] = json!("inference_deadline");
    }
    Ok(result)
}
