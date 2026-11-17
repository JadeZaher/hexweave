//! Next-byte choice for generation and full-window story openings; see AGENTS.md §Generation.

use super::assay_update::draw;
use super::core_corpus::END;
use super::generation::lowest_maximum;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// How generation picks each byte: greedy argmax (lowest byte on ties) or seeded nucleus sampling.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decoder {
    Greedy,
    /// Nucleus (top-p) sampling at a temperature: Holtzman et al. 2019, arXiv:1904.09751.
    Nucleus {
        temperature: f64,
        top_p: f64,
        seed: u64,
    },
}

impl Decoder {
    /// Refuse sampling outside temperature (0, 10] and top-p (0, 1].
    pub fn validate(self) -> Result<()> {
        if let Self::Nucleus {
            temperature, top_p, ..
        } = self
        {
            ensure!(
                temperature.is_finite() && temperature > 0.0 && temperature <= 10.0,
                "temperature must be in (0, 10]"
            );
            ensure!(
                top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
                "top-p must be in (0, 1]"
            );
        }
        Ok(())
    }

    /// Byte for generated position `step`: a pure function of the logits, settings, seed and step.
    pub(super) fn choose(self, logits: &[f32], step: usize) -> Result<usize> {
        match self {
            Self::Greedy => lowest_maximum(logits),
            Self::Nucleus {
                temperature,
                top_p,
                seed,
            } => nucleus(logits, temperature, top_p, draw(seed, step)),
        }
    }
}

/// Draw from the smallest most-probable set holding `top_p` of the tempered mass; `unit` is in [0, 1).
fn nucleus(logits: &[f32], temperature: f64, top_p: f64, unit: f64) -> Result<usize> {
    let peak = f64::from(logits[lowest_maximum(logits)?]);
    let weights: Vec<f64> = logits
        .iter()
        .map(|&logit| ((f64::from(logit) - peak) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let mut order: Vec<usize> = (0..weights.len()).collect();
    // Most probable first; equal weights keep byte order, so the kept set is deterministic.
    order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b)));
    let (mut kept, mut size) = (0.0, 0);
    for &index in &order {
        kept += weights[index];
        size += 1;
        if kept >= top_p * total {
            break;
        }
    }
    let mut remaining = unit * kept;
    for &index in &order[..size] {
        remaining -= weights[index];
        if remaining < 0.0 {
            return Ok(index);
        }
    }
    Ok(order[size - 1])
}

/// Held-out text ending at story boundary `slot` of `slots` (evenly spaced), then `prompt`, filling
/// one training `window` so the newest byte sits where training always puts it; see AGENTS.md §Generation.
pub(super) fn story_opening(
    heldout: &[u8],
    window: usize,
    slot: usize,
    slots: usize,
    prompt: &[u8],
) -> Result<Vec<u8>> {
    ensure!(slot < slots, "story context slot out of range");
    let ends: Vec<usize> = heldout
        .iter()
        .enumerate()
        .filter(|&(index, &byte)| byte == END && index + 1 >= window)
        .map(|(index, _)| index)
        .collect();
    ensure!(
        !ends.is_empty(),
        "held-out text has no story boundary a full window deep"
    );
    let end = ends[(slot + 1) * ends.len() / (slots + 1)];
    let fill = window.saturating_sub(prompt.len());
    Ok([&heldout[end + 1 - fill..end + 1], prompt].concat())
}
