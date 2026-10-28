//! Core-path evaluations: science dev generation and the retrieval-drill probes; see AGENTS.md §core.

use super::assay_source::Source as AssaySource;
use super::core_corpus::{window_before, Corpus, END};
use super::data::Dataset;
use super::evaluation::{self, Limits, Loss};
use super::execution::forward_path;
use super::generation::{self, Terminal};
use super::resident::ResidentStore;
use super::state::State;
use super::tuning;
use super::tuning_measure;
use anyhow::Result;
use serde_json::{json, Value};

const HELDOUT_SPANS: usize = 16;
const HELDOUT_SPAN: usize = 128;
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
/// as `alternate`, the other), plus greedy continuations of story openings in the run's mode.
pub(super) fn corpus(state: &State, store: &mut ResidentStore, corpus: &Corpus) -> Result<Value> {
    let float = tuning::current().float_forward;
    let (own, targets) = heldout_nll(state, store, corpus)?;
    tuning::set_float_forward(!float);
    let alternate = heldout_nll(state, store, corpus);
    tuning::set_float_forward(float);
    let (alternate, _) = alternate?;
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
    }))
}

/// Mean held-out NLL (nats per byte) over fixed spans under the installed forward mode.
fn heldout_nll(state: &State, store: &mut ResidentStore, corpus: &Corpus) -> Result<(f64, u64)> {
    let heldout = &corpus.heldout;
    let stride = heldout.len() / HELDOUT_SPANS;
    let (mut nll, mut targets) = (0.0f64, 0u64);
    for span in 0..HELDOUT_SPANS {
        for t in span * stride..(span * stride + HELDOUT_SPAN).min(heldout.len()) {
            let prefix = state
                .shared
                .prefix(&window_before(heldout, t, corpus.window))?;
            let width = state.geometry.width;
            let trace = forward_path(
                store,
                state.geometry,
                &prefix,
                vec![0.0; width],
                &mut None,
                state.tile_versions,
            )?;
            nll += f64::from(state.shared.readout(&trace.output)?.loss(heldout[t])?);
            targets += 1;
        }
    }
    Ok((nll / targets.max(1) as f64, targets))
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
