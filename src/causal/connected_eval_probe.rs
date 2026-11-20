//! Fixed-offset frozen probes and explicit score counts; see CONNECTED.md.

use super::super::connected_sections_read::{cosine, entry};
use super::super::connected_state::State;
use super::super::connected_store::Store;
use super::super::core_corpus::{window_before, Corpus, SPAN_BYTES};
use super::super::geometry::TILES;
use super::super::store_api::TileStore;
use super::replay::Replay;
use super::Budget;
use anyhow::Result;
use serde::Serialize;
use serde_json::{json, Value};

pub(super) const RETENTION_OFFSETS: [u64; 6] = [0, 1024, 4096, 16384, 65536, 262144];

pub(super) struct Probe {
    offset: u64,
    bytes: Vec<u8>,
    first: usize,
    scored: usize,
    first_observed_at: Option<u64>,
}

#[derive(Serialize)]
struct Score {
    byte_offset: u64,
    requested_targets: usize,
    scored_targets: usize,
    nll_sum: f64,
    bits_per_byte: Option<f64>,
    partial: bool,
    first_pass_observed_at_anchor: Option<u64>,
    anchors_since_first_pass: Option<u64>,
}

pub(super) fn heldout(corpus: &Corpus, spans: usize) -> Vec<Probe> {
    (0..spans)
        .map(|index| {
            let begin = index * corpus.heldout.len() / spans;
            let left = begin.saturating_sub(corpus.window + 2);
            let end = (begin + SPAN_BYTES).min(corpus.heldout.len());
            Probe {
                offset: begin as u64,
                bytes: corpus.heldout[left..end].to_vec(),
                first: begin - left,
                scored: end - begin,
                first_observed_at: None,
            }
        })
        .collect()
}

pub(super) fn retention(
    corpus: &Corpus,
    spans: usize,
    anchors: u64,
    just_observed: bool,
) -> Result<Vec<Probe>> {
    let mut probes = Vec::new();
    for &offset in RETENTION_OFFSETS.iter().take(spans) {
        if just_observed && offset + SPAN_BYTES as u64 != anchors {
            continue;
        }
        if let Some((bytes, first)) = corpus.retention_span(offset, SPAN_BYTES, anchors)? {
            probes.push(Probe {
                offset,
                bytes,
                first,
                scored: SPAN_BYTES,
                first_observed_at: Some(offset + SPAN_BYTES as u64),
            });
        }
    }
    Ok(probes)
}

pub(super) fn score(
    state: &State,
    store: &mut Store,
    probes: &[Probe],
    window: usize,
    budget: &Budget<'_>,
) -> Result<Value> {
    score_with(state, store, probes, window, budget, true)
}

/// `score`, with the shared-section read skipped when `read` is false.
pub(super) fn score_with(
    state: &State,
    store: &mut Store,
    probes: &[Probe],
    window: usize,
    budget: &Budget<'_>,
    read: bool,
) -> Result<Value> {
    let mut scores = Vec::with_capacity(probes.len());
    let (mut loss, mut targets, mut requested) = (0.0, 0, 0);
    for probe in probes {
        let mut result = Score {
            byte_offset: probe.offset,
            requested_targets: probe.scored,
            scored_targets: 0,
            nll_sum: 0.0,
            bits_per_byte: None,
            partial: true,
            first_pass_observed_at_anchor: probe.first_observed_at,
            anchors_since_first_pass: probe.first_observed_at.map(|at| state.anchors - at),
        };
        if !budget.interrupted() {
            let mut replay = Replay::with_read(state, store, read)?;
            for t in probe.first.saturating_sub(2)..probe.first + probe.scored {
                if budget.interrupted() {
                    break;
                }
                let value =
                    replay.score(&window_before(&probe.bytes, t, window), probe.bytes[t])?;
                if t >= probe.first {
                    result.nll_sum += f64::from(value);
                    result.scored_targets += 1;
                }
            }
        }
        result.bits_per_byte = bits(result.nll_sum, result.scored_targets);
        result.partial = result.scored_targets != result.requested_targets;
        loss += result.nll_sum;
        targets += result.scored_targets;
        requested += result.requested_targets;
        scores.push(result);
    }
    Ok(
        json!({"requested_targets": requested, "scored_targets": targets,
        "nll_per_byte": (targets > 0).then_some(loss / targets.max(1) as f64),
        "bits_per_byte": bits(loss, targets), "partial": targets != requested,
        "probes": scores}),
    )
}

/// Route-entry states `tile_similarity` samples: round-robin over the probes' scored positions.
pub(super) const SIMILARITY_POSITIONS: usize = 8;

/// Mean pairwise cosine between every tile's output change (output - input) at the same post-read
/// route-entry states
/// (zero history), over up to `SIMILARITY_POSITIONS` positions; None without positions.
pub(super) fn similarity(
    state: &State,
    store: &mut Store,
    probes: &[Probe],
    window: usize,
) -> Result<Option<f64>> {
    super::super::tuning::set_connected(&state.options);
    let mut entries = Vec::new();
    for round in 0..SIMILARITY_POSITIONS {
        for probe in probes.iter().filter(|probe| round < probe.scored) {
            if entries.len() < SIMILARITY_POSITIONS {
                let bytes = window_before(&probe.bytes, probe.first + round, window);
                let prefix = state.shared.prefix(&bytes)?;
                let sections = state.sections.as_ref();
                let (input, _) = entry(sections, state.geometry, &prefix)?;
                entries.push((input, prefix.memory));
            }
        }
    }
    let history = vec![0.0; state.geometry.width];
    let mut outputs = Vec::with_capacity(TILES);
    for tile in 0..TILES {
        outputs.push(store.read(tile, |tile| {
            // The residual change: every tile's output is close to its input.
            let forward = |(input, memory): &(Vec<f32>, Vec<f32>)| -> Result<Vec<f32>> {
                let output = tile.forward(input, memory, &history)?.output;
                Ok(output.iter().zip(input).map(|(y, x)| y - x).collect())
            };
            entries.iter().map(forward).collect::<Result<Vec<_>>>()
        })?);
    }
    Ok(pairwise_cosine(&outputs))
}

/// Mean cosine over every pair of tiles at every shared position; None without a pair.
pub(super) fn pairwise_cosine(changes: &[Vec<Vec<f32>>]) -> Option<f64> {
    let (mut total, mut pairs) = (0.0, 0u64);
    for (index, first) in changes.iter().enumerate() {
        for second in &changes[index + 1..] {
            for (a, b) in first.iter().zip(second) {
                total += cosine(a, b);
                pairs += 1;
            }
        }
    }
    (pairs > 0).then(|| total / pairs as f64)
}

fn bits(loss: f64, count: usize) -> Option<f64> {
    (count > 0).then_some(loss / count.max(1) as f64 / std::f64::consts::LN_2)
}
