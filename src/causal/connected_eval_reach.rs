//! Context-reach diagnostic: held-out loss with the distant window replaced by unrelated text;
//! see CONNECTED.md §Context reach.

use super::super::connected_state::State;
use super::super::connected_store::Store;
use super::super::core_corpus::window_before;
use super::probe::Probe;
use super::replay::Replay;
use super::Budget;
use anyhow::{bail, ensure, Result};
use serde_json::{json, Value};

/// Retained context lengths below the window; the full window is always measured last.
const CONTEXTS: [usize; 10] = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256];

/// Target classes: the byte's index inside a run of ASCII letters, or any other byte.
const CLASSES: [&str; 5] = [
    "word_byte_1",
    "word_byte_2",
    "word_byte_3",
    "word_byte_4_plus",
    "non_letter",
];

/// The newest `keep` bytes of `window`, with every older position taken from `foreign` (cycled).
pub(super) fn splice(window: &[u8], foreign: &[u8], keep: usize) -> Vec<u8> {
    let cut = window.len().saturating_sub(keep);
    let mut bytes: Vec<u8> = (0..cut).map(|at| foreign[at % foreign.len()]).collect();
    bytes.extend_from_slice(&window[cut..]);
    bytes
}

/// Another span's bytes, from half the probes away onward; the first that fills a window.
fn foreign(probes: &[Probe], index: usize, window: usize) -> &[u8] {
    let count = probes.len();
    let others = (count / 2..count / 2 + count - 1).map(|step| &probes[(index + step) % count]);
    let full = others.clone().find(|probe| probe.bytes.len() >= window);
    &full
        .or_else(|| others.clone().next())
        .expect("two probes")
        .bytes
}

/// Index into `CLASSES` of the target at `t`: letters directly before it (at most 3), else 4.
pub(super) fn class(bytes: &[u8], t: usize) -> usize {
    if !bytes[t].is_ascii_alphabetic() {
        return 4;
    }
    let before = bytes[..t].iter().rev().take(3);
    before.take_while(|byte| byte.is_ascii_alphabetic()).count()
}

/// One row per retained context length; the full-window row must equal `standard` bits/byte.
pub(super) fn measure(
    state: &State,
    store: &mut Store,
    probes: &[Probe],
    window: usize,
    budget: &Budget<'_>,
    standard: Option<f64>,
) -> Result<Value> {
    ensure!(probes.len() >= 2, "context reach needs two held-out spans");
    let lengths = CONTEXTS.iter().copied().filter(|keep| *keep < window);
    let mut rows = Vec::new();
    for keep in lengths.chain([window]) {
        let (mut sums, mut counts) = ([0.0f64; 5], [0usize; 5]);
        for (index, probe) in probes.iter().enumerate() {
            let foreign = foreign(probes, index, window);
            let mut replay = Replay::new(state, store)?;
            let start = super::probe::observed_from(&replay, &probe.bytes, probe.first, window);
            for t in start..probe.first + probe.scored {
                if budget.interrupted() {
                    bail!("context reach deadline reached at context {keep}");
                }
                // The contextual-row warm-up reads true bytes; foreign rows never take a true state.
                let kept = if t + 2 < probe.first { window } else { keep };
                let bytes = splice(&window_before(&probe.bytes, t, window), foreign, kept);
                let foreign_rows = bytes.len().saturating_sub(kept);
                let value = replay.score_at(&bytes, probe.bytes[t], t as u64, foreign_rows)?;
                if t >= probe.first {
                    let class = class(&probe.bytes, t);
                    sums[class] += f64::from(value);
                    counts[class] += 1;
                }
            }
        }
        rows.push(row(keep, &sums, &counts));
    }
    let full = rows.last().and_then(|row| row["bits_per_byte"].as_f64());
    ensure!(
        matches!((full, standard), (Some(a), Some(b)) if (a - b).abs() <= 1e-6 * b.abs().max(1.0)),
        "full-window context reach differs from the held-out score"
    );
    Ok(json!({"window": window, "heldout_bits_per_byte": standard, "rows": rows}))
}

fn row(keep: usize, sums: &[f64; 5], counts: &[usize; 5]) -> Value {
    let bits = |loss: f64, count: usize| {
        (count > 0).then_some(loss / count.max(1) as f64 / std::f64::consts::LN_2)
    };
    let classes: serde_json::Map<String, Value> = CLASSES
        .iter()
        .zip(sums.iter().zip(counts))
        .map(|(name, (loss, count))| {
            let value = json!({"targets": count, "bits_per_byte": bits(*loss, *count)});
            (name.to_string(), value)
        })
        .collect();
    let (loss, count) = (sums.iter().sum(), counts.iter().sum());
    json!({"context_bytes": keep, "targets": count, "bits_per_byte": bits(loss, count),
        "classes": classes})
}

#[cfg(test)]
mod tests {
    use super::{class, splice};

    #[test]
    fn splice_keeps_only_the_newest_bytes_and_classes_follow_word_position() {
        assert_eq!(splice(b"abcdef", b"XY", 2), b"XYXYef");
        assert_eq!(splice(b"abcdef", b"XY", 9), b"abcdef");
        let text = b"a girls. x";
        let classes: Vec<usize> = (0..text.len()).map(|t| class(text, t)).collect();
        assert_eq!(classes, [0, 4, 0, 1, 2, 3, 3, 4, 4, 0]);
    }
}
