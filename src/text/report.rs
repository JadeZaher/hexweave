use super::data::Corpus;
use super::execution::Counts;
use super::history::History;
use super::node::parameter_count;
use super::shared::Shared;
use super::storage::Store;
use super::{CreditDiagnostics, Options, CONTEXT, DEPTH, FEATURES, TILES, VOCAB, WIDTH};
use anyhow::Result;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Default, Serialize)]
pub(super) struct Timings {
    pub initialization_seconds: f64,
    pub initial_evaluation_seconds: f64,
    pub training_seconds: f64,
    pub final_evaluation_seconds: f64,
    pub snapshot_seconds: f64,
    pub total_seconds: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn report(
    options: &Options,
    data: &Corpus,
    store: &Store,
    shared: &Shared,
    history: &History,
    counts: &Counts,
    timings: &Timings,
    completed: usize,
    stopped: bool,
    initial_loss: f32,
    final_loss: Option<f32>,
    evaluation_examples: usize,
) -> Result<Value> {
    let tile_values: usize = (0..TILES)
        .map(|id| parameter_count(store.mesh.neighbors(id).unwrap().len() + 1))
        .sum();
    let max_tile_values = parameter_count(7);
    let gradient_bundle_bytes = max_tile_values * std::mem::size_of::<f32>();
    let shadow_bundles = usize::from(options.credit_diagnostics == CreditDiagnostics::On) * 3;
    Ok(json!({
        "protocol": "text-pulse-v4", "status": if stopped { "stopped" } else { "complete" },
        "resolved_kernel": crate::ternary::kernel_name()?,
        "options": options, "completed_step_opportunities": completed, "data": data.identity(),
        "architecture": {"width": WIDTH, "vocabulary": VOCAB, "context": CONTEXT,
            "tiles": TILES, "maximum_path_visits": DEPTH, "parameter_count": tile_values + 2 * WIDTH * VOCAB + WIDTH * FEATURES,
            "routing": "learned hard hex neighbors plus halt; revisits legal", "activation": "learned ternary three-knot continuous piecewise-linear curve",
            "context_encoder": "fixed position-weighted byte embedding sum; no attention", "floating_weight_masters": false,
            "training_pulse": {"kind": "joint-tile-pulse-v1", "context_inputs": 2 * WIDTH,
                "latent_width": WIDTH, "heads": ["transform_gain", "curve_shift", "route_bias"],
                "all_coefficients_ternary": true, "route_credit": options.route_credit,
                "scope": "one learned trunk per tile; correction predictor remains separately shared"}},
        "evaluation": {"initial_mean_nll_nats_per_byte": initial_loss, "final_mean_nll_nats_per_byte": final_loss,
            "examples_per_pass": evaluation_examples, "history_reset_each_example": true, "training_mutation_checked": true,
            "quality_claim": false},
        "counts": counts, "shared_transitions": {"embedding": shared.transitions[0], "readout": shared.transitions[1], "predictor": shared.transitions[2]},
        "credit_diagnostics": counts.credit.as_ref().map(|value| value.report()),
        "predictor_error": {"sum_per_update_mean_squared_error": counts.predictor_error_sum,
            "update_denominator": counts.predictor_updates,
            "mean_squared_error": if counts.predictor_updates > 0 { Some(counts.predictor_error_sum / counts.predictor_updates as f32) } else { None },
            "includes_replay": true, "components_per_update": WIDTH},
        "tile_transitions": {"transform": counts.transitions[0], "curve": counts.transitions[1], "router": counts.transitions[2],
            "pulse_context": counts.transitions[3], "pulse_heads": counts.transitions[4]},
        "disk_io": store.io, "timings": timings,
        "residency": {"live_tile_limit": 1, "peak_live_tiles": store.io.peak_live_tiles,
            "peak_live_tiles_basis": "structural ownership bound; not a sampled allocator or RSS measurement",
            "shared_packed_weight_bytes": shared.packed_bytes(), "shared_evidence_bytes": shared.evidence_bytes(),
            "maximum_live_tile_packed_bytes": max_tile_values.div_ceil(4),
            "maximum_live_tile_evidence_bytes": max_tile_values * 2,
            "tile_evidence_scope": "separate conservative apply-evidence payload bound; excluded from gradient bundle total",
            "maximum_actual_update_gradient_bundle_bytes": gradient_bundle_bytes,
            "diagnostic_shadow_gradient_bundles": shadow_bundles,
            "maximum_diagnostic_shadow_gradient_bytes": shadow_bundles * gradient_bundle_bytes,
            "maximum_concurrent_gradient_bundle_bytes": (1 + shadow_bundles) * gradient_bundle_bytes,
            "gradient_bundle_scope": "largest tile actual-update and enabled full shadow bundles only; excludes other scratch and evidence",
            "history_record_limit": 8, "history_records_peak": history.peak,
            "history_stale_rejections": history.rejected,
            "physical_rss_measured_here": false, "inactive_tiles_application_loaded": false,
            "os_file_cache_controlled": false, "serialized_current_tile_copies_exist": true,
            "ledger_scope": "selected payload bounds only; excludes containers, other temporary scratch vectors, activations, serialization and runtime; not complete heap or RSS"},
        "artifact_bounds": {"maximum_versioned_tile_files": TILES + completed * DEPTH,
            "maximum_tile_file_bytes": 8192, "checkpoint_resumable": false},
        "timing_scope": "total includes initialization, evaluation, training, enabled diagnostics and snapshot; excludes final result serialization/flush; incremental diagnostic overhead is not isolated",
        "training_target_tokens_per_second": if timings.training_seconds > 0.0 { Some(counts.anchor_targets as f64 / timings.training_seconds) } else { None },
        "limitations": ["tiny authored byte task, not full F or attention pretraining", "threshold-one ternary evidence is experimental",
            "real correction only at anchors; predictor may be wrong", "anchor-only is periodic ablation, not dense baseline",
            "node residency is application ownership, not proof of physical page eviction", "no parallel waves, fanout, GPU or resume"]
    }))
}
