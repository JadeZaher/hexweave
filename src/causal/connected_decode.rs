//! Bounded readout diagnostics, separate from training and default decoding; see CONNECTED.md.

use super::connected_anchor::Trace;
use super::connected_signal::Memory;
use super::decoding::Decoder;
use super::generation::lowest_maximum;
use super::geometry::VOCAB;
use super::shared::{Prefix, Readout, Shared};
use super::{norm, tuning};
use anyhow::{ensure, Result};
use serde_json::{json, Value};

pub(super) const POLICY: &str = "connected-full-window-frozen-local-lanes-v3";

#[derive(Clone, Copy)]
pub(super) struct Options {
    pub trace_bytes: usize,
    pub generated_feedback: bool,
    pub quantize_audit: bool,
    pub decoder: Decoder,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            trace_bytes: 0,
            generated_feedback: true,
            quantize_audit: false,
            decoder: Decoder::Greedy,
        }
    }
}

impl Options {
    pub fn validate(self) -> Result<()> {
        self.decoder.validate()?;
        ensure!(self.trace_bytes <= 16, "trace-bytes must be in 0..=16");
        ensure!(
            self.generated_feedback || self.trace_bytes > 0,
            "prompt-only feedback requires a nonzero trace-bytes diagnostic"
        );
        Ok(())
    }
}

pub(super) fn trace(
    shared: &Shared,
    prefix: &Prefix,
    route: &Trace,
    readout: &Readout,
    selected: u8,
    memory: &Memory,
) -> Result<Value> {
    norm::vector(&readout.logits, VOCAB)?;
    ensure!(
        usize::from(selected) == lowest_maximum(&readout.logits)?,
        "decoder choice differs from logits"
    );
    let mut ranks = (0..VOCAB).collect::<Vec<_>>();
    ranks.sort_by(|&a, &b| {
        readout.logits[b]
            .partial_cmp(&readout.logits[a])
            .unwrap()
            .then(a.cmp(&b))
    });
    let mut probabilities = readout.logits.clone();
    norm::softmax(&mut probabilities)?;
    let top = ranks[..5]
        .iter()
        .map(|&byte| {
            json!({"byte":byte,
        "logit":readout.logits[byte],"probability":probabilities[byte]})
        })
        .collect::<Vec<_>>();
    let scalar = scalar_logits(shared, &readout.normalized.values)?;
    let error = scalar
        .iter()
        .zip(&readout.logits)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    let maximum = readout.logits[usize::from(selected)];
    Ok(json!({
        "selected_byte": selected, "selected_probability": probabilities[usize::from(selected)],
        "logits": readout.logits, "top": top,
        "runner_up_margin": maximum - readout.logits[ranks[1]],
        "maximum_ties": readout.logits.iter().filter(|&&value| value == maximum).count(),
        "space_rank": ranks.iter().position(|&byte| byte == usize::from(b' ')).unwrap() + 1,
        "space_gap_from_best": maximum - readout.logits[usize::from(b' ')],
        "end_probability": probabilities[usize::from(super::core_corpus::END)],
        "entropy_bits": -probabilities.iter().filter(|&&p| p > 0.0).map(|&p| f64::from(p)*f64::from(p).log2()).sum::<f64>(),
        "visible_prefix_bytes": prefix.bytes, "visible_prefix_length": prefix.bytes.len(),
        "route_tiles": route.frames.iter().map(|frame| frame.tile).collect::<Vec<_>>(),
        "memory_records": memory.records.len(), "memory_publications": memory.publications,
        "observed_publications": memory.observed, "predicted_publications": memory.predicted,
        "self_generated_publications": memory.self_generated,
        "scalar_readout_max_abs_error": error, "scalar_selected_byte": lowest_maximum(&scalar)?,
    }))
}

fn scalar_logits(shared: &Shared, normalized: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !tuning::current().float_forward,
        "connected scalar readout requires ternary weights"
    );
    let scale = (shared.geometry.width as f32).sqrt().recip() / tuning::current().temperature;
    let mut result = vec![0.0; VOCAB];
    for (byte, value) in result.iter_mut().enumerate() {
        for (row, &input) in normalized.iter().enumerate() {
            let weight =
                match super::connected_precision::matrix(&shared.weights, 0, shared.version)? {
                    Some(weights) => weights[row * VOCAB + byte],
                    None => f32::from(shared.embedding.value(row * VOCAB + byte)?),
                };
            *value += weight * input;
        }
        *value *= scale;
        if let Some(bias) = &shared.bias {
            *value += bias[byte];
        }
    }
    norm::finite(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_scalar_readout_agrees_with_packed_kernel_on_nonzero_logits() -> Result<()> {
        let _guard = tuning::scoped(tuning::Knobs::core(0.003, 7));
        let shared = Shared::new(super::super::geometry::Geometry::fixture(), 17)?;
        let input = (0..shared.geometry.width)
            .map(|i| (i as f32 + 0.5).sin())
            .collect::<Vec<_>>();
        let readout = shared.readout(&input)?;
        let scalar = scalar_logits(&shared, &readout.normalized.values)?;
        assert!(readout.logits.iter().any(|value| value.abs() > 0.1));
        for (actual, expected) in readout.logits.iter().zip(&scalar) {
            assert!((*actual - *expected).abs() < 1e-6);
        }
        assert_eq!(lowest_maximum(&readout.logits)?, lowest_maximum(&scalar)?);
        Ok(())
    }
}
