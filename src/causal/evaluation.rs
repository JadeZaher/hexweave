use super::data::{DataIdentity, Dataset};
use super::data_contracts::{Split, TargetKind};
use super::execution::forward_path;
use super::generation::{self, Generation, Reason, Terminal};
use super::geometry::Geometry;
use super::immutable;
use super::state::{count, State};
use super::storage_paths::digest;
use super::store_api::TileStore;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Loss {
    pub targets: u64,
    pub nll_sum: f64,
}
impl Loss {
    fn add(&mut self, value: f32) -> Result<()> {
        ensure!(value.is_finite() && value >= 0.0, "invalid evaluation loss");
        count(&mut self.targets, 1)?;
        self.nll_sum += f64::from(value);
        ensure!(self.nll_sum.is_finite(), "evaluation loss sum overflow");
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Losses {
    pub total: Loss,
    pub answer: Loss,
    pub citation: Loss,
    pub format: Loss,
}
impl Losses {
    fn add(&mut self, value: f32, kind: TargetKind) -> Result<()> {
        self.total.add(value)?;
        match kind {
            TargetKind::Answer => &mut self.answer,
            TargetKind::Citation => &mut self.citation,
            TargetKind::Format => &mut self.format,
        }
        .add(value)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "status")]
pub(super) enum Coverage {
    Complete,
    Limit,
    Interrupted(Terminal),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub example_id: String,
    pub prompt_sha256: String,
    pub evidence_sha256: String,
    pub expected_unknown: bool,
    pub valid_json: bool,
    pub exact_answer: bool,
    pub exact_citation: bool,
    pub exact_completion: bool,
    pub unsupported_answer: bool,
    pub generation: Generation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Evaluation {
    pub dataset: DataIdentity,
    pub before_sha256: String,
    pub after_sha256: String,
    pub teacher_forced: Losses,
    pub expected_targets: u64,
    pub complete_rows: usize,
    pub teacher_coverage: Coverage,
    pub cases: Vec<Case>,
    pub generation_coverage: Coverage,
    pub wall_millis: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Limits {
    pub targets: u64,
    pub cases: usize,
    pub bytes_per_case: usize,
}

pub(super) fn evaluate<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    dataset: &Dataset,
    limits: Limits,
    poll: &mut F,
) -> Result<Evaluation> {
    ensure!(
        state.geometry == Geometry::candidate(),
        "real QA requires context1024 candidate"
    );
    ensure!(
        dataset.identity().split == Split::Development,
        "read-only development partition required"
    );
    ensure!(
        (1..=65536).contains(&limits.targets)
            && (1..=32).contains(&limits.cases)
            && (1..=192).contains(&limits.bytes_per_case),
        "invalid evaluation limits"
    );
    let started = Instant::now();
    let before_sha256 = immutable::identity(state, store)?;
    let (teacher_forced, complete_rows, teacher_coverage) =
        teacher(state, store, dataset, limits.targets, poll)?;
    let (cases, generation_coverage) = generated(state, store, dataset, limits, poll)?;
    let after_sha256 = immutable::identity(state, store)?;
    ensure!(
        before_sha256 == after_sha256,
        "evaluation mutated model or continuation state"
    );
    Ok(Evaluation {
        dataset: dataset.identity().clone(),
        before_sha256,
        after_sha256,
        teacher_forced,
        expected_targets: dataset.identity().completion_targets_per_epoch,
        complete_rows,
        teacher_coverage,
        cases,
        generation_coverage,
        wall_millis: u64::try_from(started.elapsed().as_millis())?,
    })
}

fn teacher<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    dataset: &Dataset,
    maximum: u64,
    poll: &mut F,
) -> Result<(Losses, usize, Coverage)> {
    let mut losses = Losses::default();
    for row_index in 0..dataset.len() {
        let row = dataset.record(row_index)?;
        let mut visible = row.prompt;
        for (offset, &target) in row.completion.iter().enumerate() {
            if let Some(status) = poll()? {
                return Ok((losses, row_index, Coverage::Interrupted(status)));
            }
            if losses.total.targets == maximum {
                return Ok((losses, row_index, Coverage::Limit));
            }
            let prefix = state.shared.prefix(&visible)?;
            let trace = forward_path(
                store,
                state.geometry,
                &prefix,
                vec![0.0; state.geometry.width],
                &mut None,
                state.tile_versions,
            )?;
            let loss = state.shared.readout(&trace.output)?.loss(target)?;
            losses.add(loss, row.masks.kind(offset))?;
            visible.push(target);
        }
    }
    Ok((losses, dataset.len(), Coverage::Complete))
}

fn generated<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    dataset: &Dataset,
    limits: Limits,
    poll: &mut F,
) -> Result<(Vec<Case>, Coverage)> {
    let mut cases = Vec::with_capacity(dataset.len().min(limits.cases));
    for index in 0..dataset.len().min(limits.cases) {
        if let Some(status) = poll()? {
            return Ok((cases, Coverage::Interrupted(status)));
        }
        let row = dataset.record(index)?;
        let generation =
            generation::generate(state, store, &row.prompt, limits.bytes_per_case, poll)?;
        let interrupted = match &generation.stop_reason {
            Reason::Interrupted(status) => Some(*status),
            _ => None,
        };
        let expected: serde_json::Value = serde_json::from_slice(&row.completion)?;
        let actual = parse_answer(&generation.raw_bytes);
        let expected_unknown = expected["answer"] == "UNKNOWN";
        cases.push(Case {
            example_id: row.id,
            prompt_sha256: digest(&row.prompt),
            evidence_sha256: row.evidence_sha256,
            expected_unknown,
            valid_json: actual.is_some(),
            exact_answer: actual
                .as_ref()
                .is_some_and(|value| value["answer"] == expected["answer"]),
            exact_citation: actual
                .as_ref()
                .is_some_and(|value| value["citation"] == expected["citation"]),
            exact_completion: generation.raw_bytes == row.completion,
            unsupported_answer: expected_unknown
                && actual
                    .as_ref()
                    .is_some_and(|value| value["answer"] != "UNKNOWN"),
            generation,
        });
        if let Some(status) = interrupted {
            return Ok((cases, Coverage::Interrupted(status)));
        }
    }
    Ok((
        cases,
        if limits.cases >= dataset.len() {
            Coverage::Complete
        } else {
            Coverage::Limit
        },
    ))
}

pub(super) fn parse_answer(bytes: &[u8]) -> Option<serde_json::Value> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Answer {
        answer: String,
        citation: serde_json::Value,
    }
    let parsed: Answer = serde_json::from_slice(bytes).ok()?;
    if !parsed.citation.is_null() && !parsed.citation.is_string() {
        return None;
    }
    let _ = parsed.answer;
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let object = value.as_object()?;
    if object.len() != 2 || !object.get("answer")?.is_string() {
        return None;
    }
    let citation = object.get("citation")?;
    if !citation.is_null() && !citation.is_string() {
        return None;
    }
    Some(value)
}
