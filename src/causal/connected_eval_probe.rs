//! Fixed-offset frozen probes and explicit score counts; see CONNECTED.md.

use super::super::connected_state::State;
use super::super::connected_store::Store;
use super::super::core_corpus::{window_before, Corpus, SPAN_BYTES};
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
            let mut replay = Replay::new(state, store)?;
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

fn bits(loss: f64, count: usize) -> Option<f64> {
    (count > 0).then_some(loss / count.max(1) as f64 / std::f64::consts::LN_2)
}
