//! Retrieval measurements over the assay rows (core drill eval and harness); see AGENTS.md §tuning.

use super::assay_measure::measure;
use super::assay_source::{Row, Source};
use super::execution::forward_path;
use super::generation::Terminal;
use super::geometry::VISITS;
use super::state::State;
use super::store_api::TileStore;
use super::tuning_sources::{swap, WORDS};
use anyhow::Result;
use serde_json::{json, Value};

/// Teacher-forced log-probability of each candidate word plus LF (multiple-choice probe).
fn choices<S: TileStore>(state: &State, store: &mut S, row: &Row) -> Result<[f32; 4]> {
    let mut scores = [0.0; 4];
    for (score, word) in scores.iter_mut().zip(WORDS) {
        let mut visible = row.prompt.clone();
        for &byte in word.iter().chain(b"\n") {
            let prefix = state.shared.prefix(&visible)?;
            let history = vec![0.0; state.geometry.width];
            let versions = state.tile_versions;
            let trace = forward_path(store, state.geometry, &prefix, history, &mut None, versions)?;
            *score -= state.shared.readout(&trace.output)?.loss(byte)?;
            visible.push(byte);
        }
    }
    Ok(scores)
}

/// Foundation attention mass at the answer position on the asked key (48), slot A (12-15),
/// slot B (22-25) and the source labels (8, 18), as [visit][head][region], averaged over the rows
/// that reached each visit; `reach` counts those rows (routes may halt early).
pub(super) fn attention_mass<S: TileStore>(
    state: &State,
    store: &mut S,
    source: &Source,
) -> Result<Value> {
    let regions = [
        vec![48],
        (12..16).collect(),
        (22..26).collect(),
        vec![8, 18],
    ];
    let (width, heads) = (state.geometry.width, state.geometry.heads);
    let mut mass = vec![vec![vec![0.0f64; regions.len()]; heads]; VISITS];
    let mut reach = [0usize; VISITS];
    let rows: Vec<_> = source
        .observations()?
        .into_iter()
        .filter(|row| row.group == "foundation")
        .collect();
    for row in &rows {
        let prefix = state.shared.prefix(&row.prompt)?;
        let history = vec![0.0; width];
        let versions = state.tile_versions;
        let trace = forward_path(store, state.geometry, &prefix, history, &mut None, versions)?;
        let length = prefix.memory.len() / width;
        for (visit, frame) in trace.frames.iter().enumerate() {
            reach[visit] += 1;
            let weights = store.read(frame.tile_id, |tile| {
                Ok(super::attention::forward(tile, &frame.input, &prefix.memory)?.probabilities)
            })?;
            for (head, per_head) in mass[visit].iter_mut().enumerate() {
                for (total, region) in per_head.iter_mut().zip(&regions) {
                    *total += region
                        .iter()
                        .map(|&p| f64::from(weights[head * length + p]))
                        .sum::<f64>();
                }
            }
        }
    }
    for (per_visit, &count) in mass.iter_mut().zip(&reach) {
        for total in per_visit.iter_mut().flatten() {
            *total /= count.max(1) as f64;
        }
    }
    Ok(json!({ "mass": mass, "reach": reach }))
}

/// Production `measure` per observation row plus multiple choice, and the key-swap probe.
pub(super) fn evaluate<S: TileStore>(
    state: &State,
    store: &mut S,
    source: &Source,
) -> Result<Value> {
    let mut never = || -> Result<Option<Terminal>> { Ok(None) };
    let mut cases = Vec::with_capacity(80);
    for row in source.observations()? {
        let case = measure(state, store, row, true, &mut never)?;
        cases.push(json!({
            "id": row.id, "content_nll": case.content_nll, "lf_nll": case.lf_nll,
            "first_byte_rank": case.first_byte_rank, "generated": case.generated,
            "exact": case.exact, "choice_logprob": choices(state, store, row)?,
        }));
        if let Some(swapped) = swap(row) {
            let case = measure(state, store, &swapped, true, &mut never)?;
            cases.push(json!({
                "id": swapped.id, "generated": case.generated,
                "choice_logprob": choices(state, store, &swapped)?,
            }));
        }
    }
    Ok(Value::Array(cases))
}
