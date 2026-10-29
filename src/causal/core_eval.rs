//! Core-path evaluations: science dev generation and the retrieval-drill probes; see AGENTS.md §core.

use super::assay_source::Source as AssaySource;
use super::core_corpus::{window_before, Corpus, END, SPAN_BYTES, SPAN_WARMUP};
use super::data::Dataset;
use super::evaluation::{self, Limits, Loss};
use super::execution::forward_path;
use super::generation::{self, Terminal};
use super::replay::Replay;
use super::resident::ResidentStore;
use super::state::State;
use super::tuning;
use super::tuning_measure;
use anyhow::Result;
use serde_json::{json, Value};

/// Held-out probe spans; `SPAN_BYTES`/`SPAN_WARMUP` (shared with the frozen-train probes) live in
/// `core_corpus` beside the file reads they also size.
const HELDOUT_SPANS: usize = 16;
const STORY_PROMPTS: [&str; 4] = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "Tom and his dog went to the",
    "The sun was",
];

fn per_byte(loss: &Loss) -> f64 {
    loss.nll_sum / loss.targets.max(1) as f64
}

/// Held-out bits per byte under both forward modes on the same weights (the run's own mode and,
/// as `alternate`, the other), the frozen-train generalization-gap probe in the run's mode only,
/// plus greedy continuations of story openings in the run's mode.
pub(super) fn corpus(state: &State, store: &mut ResidentStore, corpus: &Corpus) -> Result<Value> {
    let float = tuning::current().float_forward;
    let (own, targets) = heldout_nll(state, store, corpus)?;
    tuning::set_float_forward(!float);
    let alternate = heldout_nll(state, store, corpus);
    tuning::set_float_forward(float);
    let (alternate, _) = alternate?;
    let (train_frozen, train_frozen_targets) = train_frozen_nll(state, store, corpus)?;
    let mut never = || -> Result<Option<Terminal>> { Ok(None) };
    let mut samples = Vec::with_capacity(STORY_PROMPTS.len());
    for prompt in STORY_PROMPTS {
        let opening = [&[END][..], prompt.as_bytes()].concat();
        let generation = generation::generate(state, store, &opening, 160, &mut never)?;
        samples.push(
            json!({ "prompt": prompt, "text": String::from_utf8_lossy(&generation.raw_bytes) }),
        );
    }
    let mode = |float: bool| if float { "float" } else { "ternary" };
    let bits = |nats: f64| nats / std::f64::consts::LN_2;
    Ok(json!({
        "dev_nll_per_byte": own, "dev_bits_per_byte": bits(own), "forward": mode(float),
        "alternate_forward": mode(!float), "alternate_bits_per_byte": bits(alternate),
        "heldout_targets": targets, "samples": samples,
        "train_frozen_nll_per_byte": train_frozen, "train_frozen_bits_per_byte": bits(train_frozen),
        "train_frozen_targets": train_frozen_targets,
    }))
}

/// Mean teacher-forced NLL (nats per byte) over self-contained spans, replayed with frozen
/// weights: the last `scored` bytes are tallied after `SPAN_WARMUP` unscored positions; bytes
/// before those are prefix context only. Shared by the held-out and frozen-train probes.
pub(super) fn score_spans(
    state: &State,
    store: &mut ResidentStore,
    spans: &[(&[u8], usize)],
    window: usize,
) -> Result<(f64, u64)> {
    let (mut nll, mut targets) = (0.0f64, 0u64);
    for &(span, scored) in spans {
        let mut replay = Replay::default();
        let warmup_end = span.len() - scored;
        for t in warmup_end.saturating_sub(SPAN_WARMUP)..span.len() {
            let prefix = state.shared.prefix(&window_before(span, t, window))?;
            let history = replay.input(state)?;
            let trace = forward_path(
                store,
                state.geometry,
                &prefix,
                history,
                &mut None,
                state.tile_versions,
            )?;
            let loss = replay.observe(state, store, &prefix, &trace, span[t])?;
            if t >= warmup_end {
                nll += f64::from(loss);
                targets += 1;
            }
        }
    }
    Ok((nll / targets.max(1) as f64, targets))
}

/// Mean held-out NLL (nats per byte) over fixed spans under the installed forward mode.
fn heldout_nll(state: &State, store: &mut ResidentStore, corpus: &Corpus) -> Result<(f64, u64)> {
    let heldout = &corpus.heldout;
    let stride = heldout.len() / HELDOUT_SPANS;
    let bounds: Vec<(usize, usize)> = (0..HELDOUT_SPANS)
        .map(|span| {
            (
                span * stride,
                (span * stride + SPAN_BYTES).min(heldout.len()),
            )
        })
        .collect();
    let spans: Vec<(&[u8], usize)> = bounds
        .iter()
        .map(|&(begin, end)| {
            let left = begin
                .saturating_sub(SPAN_WARMUP)
                .saturating_sub(corpus.window);
            (&heldout[left..end], end - begin)
        })
        .collect();
    score_spans(state, store, &spans, corpus.window)
}

/// Mean frozen-weight NLL (nats per byte) on the fixed train spans (`Corpus::open`), run's own
/// forward mode only: a generalization-gap monitor against the held-out NLL above.
fn train_frozen_nll(
    state: &State,
    store: &mut ResidentStore,
    corpus: &Corpus,
) -> Result<(f64, u64)> {
    let spans: Vec<(&[u8], usize)> = corpus
        .train_frozen
        .iter()
        .map(|(span, scored)| (span.as_slice(), *scored))
        .collect();
    score_spans(state, store, &spans, corpus.window)
}

/// Public dev-set assessment (teacher-forced NLL plus greedy JSON answers) and train samples.
pub(super) fn science(
    state: &State,
    store: &mut ResidentStore,
    train: &Dataset,
    dev: &Dataset,
    cases: usize,
) -> Result<Value> {
    let mut never = || -> Result<Option<Terminal>> { Ok(None) };
    let limits = Limits {
        targets: 65_536,
        cases,
        bytes_per_case: 192,
    };
    let report = evaluation::evaluate(state, store, dev, limits, &mut never)?;
    let dev_cases: Vec<Value> = report
        .cases
        .iter()
        .map(|case| {
            json!({
                "id": case.example_id, "valid_json": case.valid_json,
                "exact_answer": case.exact_answer, "exact_citation": case.exact_citation,
                "expected_unknown": case.expected_unknown,
                "text": String::from_utf8_lossy(&case.generation.raw_bytes),
            })
        })
        .collect();
    let mut train_samples = Vec::new();
    for index in 0..train.len().min(4) {
        let row = train.record(index)?;
        let budget = 192.min(state.geometry.context.saturating_sub(row.prompt.len()));
        let generation = generation::generate(state, store, &row.prompt, budget, &mut never)?;
        train_samples.push(json!({
            "id": row.id, "exact_completion": generation.raw_bytes == row.completion,
            "text": String::from_utf8_lossy(&generation.raw_bytes),
            "expected": String::from_utf8_lossy(&row.completion),
        }));
    }
    let tally =
        |field: fn(&evaluation::Case) -> bool| report.cases.iter().filter(|c| field(c)).count();
    Ok(json!({
        "dev_nll_per_byte": per_byte(&report.teacher_forced.total),
        "dev_answer_nll_per_byte": per_byte(&report.teacher_forced.answer),
        "dev_citation_nll_per_byte": per_byte(&report.teacher_forced.citation),
        "dev_format_nll_per_byte": per_byte(&report.teacher_forced.format),
        "dev_cases_scored": report.cases.len(),
        "dev_complete_rows": report.complete_rows,
        "dev_expected_targets": report.expected_targets,
        "dev_teacher_coverage": serde_json::to_value(&report.teacher_coverage)?,
        "dev_generation_coverage": serde_json::to_value(&report.generation_coverage)?,
        "dev_valid_json": tally(|c| c.valid_json),
        "dev_exact_answer": tally(|c| c.exact_answer),
        "dev_exact_citation": tally(|c| c.exact_citation),
        "dev_cases": dev_cases,
        "train_samples": train_samples,
    }))
}

/// Retrieval-drill probes on the fixed assay rows: production measure, key swap, attention.
pub(super) fn drill(
    state: &State,
    store: &mut ResidentStore,
    assay: &AssaySource,
) -> Result<Value> {
    Ok(json!({
        "cases": tuning_measure::evaluate(state, store, assay)?,
        "attention": tuning_measure::attention_mass(state, store, assay)?,
    }))
}
