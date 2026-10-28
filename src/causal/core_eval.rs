//! Core-path evaluations: science dev generation, retrieval-drill probes and best-checkpoint
//! selection; see AGENTS.md §core.

use super::assay_source::Source as AssaySource;
use super::core_checkpoint;
use super::data::Dataset;
use super::evaluation::{self, Limits, Loss};
use super::generation::{self, Terminal};
use super::resident::ResidentStore;
use super::state::State;
use super::tuning_measure;
use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;

fn per_byte(loss: &Loss) -> f64 {
    loss.nll_sum / loss.targets.max(1) as f64
}

/// Lowest dev NLL in `out/best/selection.json`, or infinity before any science selection.
pub(super) fn best_so_far(out: &Path) -> f64 {
    std::fs::read(out.join("best").join("selection.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|selection| selection["dev_nll_per_byte"].as_f64())
        .unwrap_or(f64::INFINITY)
}

/// Keep the lowest-dev-NLL science state as a checkpoint under `out/best`, with its score.
pub(super) fn keep_best(
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
    if state.global_opportunity == 0 || !score.is_finite() || score >= *best {
        return Ok(());
    }
    let root = out.join("best");
    core_checkpoint::save(&root, state, store, identity)?;
    let selection = json!({ "anchors": state.global_opportunity, "dev_nll_per_byte": score });
    std::fs::write(
        root.join("selection.json"),
        serde_json::to_vec_pretty(&selection)?,
    )?;
    *best = score;
    Ok(())
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
