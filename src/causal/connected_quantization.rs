//! Non-mutating quantized-weight comparisons from FP32 checkpoints; see CONNECTED.md.

use super::connected_decode::Options;
use super::connected_precision::{Precision, View};
use super::connected_state::State;
use super::connected_store::Store;
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

#[allow(clippy::too_many_arguments)]
pub(super) fn compare(
    state: &State,
    store: &mut Store,
    context: &[u8],
    prompt: &str,
    window: usize,
    limit: usize,
    started: Instant,
    timeout: Duration,
    options: Options,
) -> Result<Value> {
    ensure!(
        state.options.precision == Precision::Fp32 && store.precision == Precision::Fp32,
        "quantize-audit requires a connected FP32 checkpoint"
    );
    let mut variants = Vec::new();
    for (index, view) in [View::Fp32, View::Int8, View::Ternary]
        .into_iter()
        .enumerate()
    {
        store.set_view(view)?;
        let quantization = summary(state, store, view)?;
        let deadline = timeout.mul_f64((index + 1) as f64 / 3.0);
        let mut result = super::connected_eval::generate_frozen_with_options(
            state, store, context, prompt, window, limit, started, deadline, options,
        )?;
        result["weight_view"] = json!(view);
        result["quantization"] = quantization;
        variants.push(result);
    }
    store.set_view(View::Fp32)?;
    Ok(json!({"quantization_audit": true,
        "execution": "fp32-arithmetic-over-frozen-rounded-weight-views",
        "trained_quantized_model": false, "master_weights_preserved": true,
        "predictor_not_quantized": true, "activation_arithmetic_fp32": true,
        "quantized_scope": "tile-matrices-and-shared-embedding-only",
        "variants": variants}))
}

fn summary(state: &State, store: &mut Store, view: View) -> Result<Value> {
    let mut matrices = Vec::new();
    for id in 0..super::geometry::TILES {
        let lease = store.checkout(id)?;
        let weights = lease
            .tile()
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing diagnostic view"))?;
        for (matrix, rounded) in weights.matrices.iter().enumerate() {
            let original = &lease.latents()[matrix].1.weights;
            matrices.push(distortion(
                json!({"tile": id, "matrix": matrix}),
                original,
                rounded,
                view,
            ));
        }
        store.return_read(lease)?;
    }
    let original = &state
        .shared_latent
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing shared master"))?
        .weights;
    let rounded = super::connected_precision::rounded(original, view)?;
    matrices.push(distortion(
        json!({"shared_embedding": true}),
        original,
        &rounded,
        view,
    ));
    Ok(
        json!({"granularity": "per_matrix", "shared_embedding_included": true,
        "scale_formula": match view {
            View::Fp32 => "identity",
            View::Int8 => "max(abs(master))/127; round(master/scale) clamped to [-127,127]; dequantize",
            View::Ternary => "mean(abs(master)); round(master/scale) clamped to [-1,1]; dequantize",
        }, "rounding": "nearest-ties-away-from-zero", "zero_scale": "all-zero-view",
        "matrices": matrices}),
    )
}

fn distortion(mut value: Value, original: &[f32], rounded: &[f32], view: View) -> Value {
    let errors = original
        .iter()
        .zip(rounded)
        .map(|(a, b)| f64::from(*a) - f64::from(*b))
        .collect::<Vec<_>>();
    value["coefficients"] = json!(original.len());
    value["mse"] = json!(errors.iter().map(|v| v * v).sum::<f64>() / original.len() as f64);
    value["max_absolute_error"] = json!(errors.iter().map(|v| v.abs()).fold(0.0f64, f64::max));
    value["scale"] = json!(super::connected_precision::scale(original, view));
    value
}
