use super::history::{History, HISTORY_CAPACITY};
use super::residency::Residents;
use super::state::Model;
use super::{Options, DEPTH, FEATURES, TILES, VOCAB, WIDTH};
use anyhow::{ensure, Result};
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Default, Serialize)]
pub(super) struct Counts {
    pub anchor_steps: usize,
    pub predicted_steps: usize,
    pub skipped_local_opportunities: usize,
    pub local_node_updates: usize,
    pub anchor_node_updates: usize,
    pub anchor_returned_frames: usize,
    pub routing_score_updates: usize,
    pub predictor_anchor_updates: usize,
    pub predictor_replay_updates: usize,
    pub predictor_forward_calls: usize,
    pub training_tile_visits: usize,
    pub evaluation_tile_visits: usize,
    pub training_router_calls: usize,
    pub evaluation_router_calls: usize,
    pub learned_halts: usize,
    pub forced_depth_exits: usize,
    pub visits_per_tile: [usize; TILES],
    pub anchor_targets_per_training_input: [usize; 12],
}

pub(super) struct Report {
    pub options: Options,
    pub initial: f32,
    pub final_loss: f32,
    pub seconds: f64,
    pub training_losses: Vec<f32>,
    pub predictor_error_sum: f32,
}

pub(super) fn summary(
    model: &Model,
    residents: &Residents,
    history: &History,
    counts: &Counts,
    report: Report,
) -> Result<Value> {
    ensure!(
        report.initial.is_finite() && report.final_loss.is_finite(),
        "nonfinite evaluation"
    );
    let parameters: usize = model.all_matrices().iter().map(|matrix| matrix.len()).sum();
    let predictor_updates = counts.predictor_anchor_updates + counts.predictor_replay_updates;
    Ok(json!({
        "protocol": "ternary-pulse-v1", "steps": report.options.steps, "seed": report.options.seed,
        "mode": report.options.mode, "resident_limit": report.options.resident_limit,
        "anchor_interval": report.options.anchor_interval, "completed_steps": report.options.steps,
        "initial_loss": report.initial, "final_loss": report.final_loss,
        "initial_heldout_loss": report.initial, "final_heldout_loss": report.final_loss,
        "seconds": report.seconds, "parameters": parameters, "all_learned_weights_ternary": true,
        "fp32_master_weights": false, "counts": counts, "anchor_losses": report.training_losses,
        "training_target_exposures": counts.anchor_steps, "evaluation_target_exposures": 8,
        "predictor_mean_training_mse": if predictor_updates > 0 {
            Some(report.predictor_error_sum / predictor_updates as f32)
        } else { None },
        "state_bytes": state_bytes(model, residents, history, report.options),
        "ternary_transitions": {
            "transform": residents.transform_transitions, "activation": residents.activation_transitions,
            "router": residents.router_transitions, "embedding": model.embedding_transitions,
            "readout": model.readout_transitions, "predictor": model.predictor_transitions
        },
        "residency": {
            "bounded_component": "optimizer evidence only; all packed weights remain in host RAM",
            "peak_nodes": residents.peak, "evidence_loads": residents.loads,
            "evidence_evictions_and_resets": residents.evictions
        },
        "history": { "capacity": HISTORY_CAPACITY, "peak_records": history.peak,
            "incompatible_or_old_drops": history.incompatible_drops, "capacity_drops": history.capacity_drops,
            "max_all_node_and_shared_weight_version_lag": 1, "max_step_age": 8 },
        "arithmetic": { "tile_transform": [WIDTH, WIDTH], "router_maximum": [7, 2 * WIDTH],
            "predictor": [WIDTH, FEATURES], "embedding": [WIDTH, VOCAB], "readout": [VOCAB, WIDTH],
            "execution": format!("{} packed-weight add/subtract/skip with f32 activations; scalar small-shape fallback, no GPU", crate::ternary::kernel_name()?),
            "ternary_update_evidence_threshold": super::THRESHOLD },
        "limits": ["synthetic heldout token IDs 12..15; training IDs 0..11; not text/pretraining quality",
            "sequential two-phase anchors, no concurrent/decentralized executor",
            "conditional-path task adjoints; router features detached in the policy gradient",
            "no fanout, attention, checkpoint/resume, full F parity, speedup or quality claim"]
    }))
}

fn state_bytes(model: &Model, residents: &Residents, history: &History, options: Options) -> Value {
    let weight_bytes: usize = model
        .all_matrices()
        .iter()
        .map(|matrix| matrix.storage_bytes())
        .sum();
    json!({
        "packed_weight_payload": weight_bytes,
        "shared_integer_evidence": model.shared_evidence_bytes(),
        "resident_integer_evidence_current": residents.current_evidence_bytes(),
        "resident_integer_evidence_peak": residents.peak_bytes,
        "resident_integer_evidence_bound": options.resident_limit * Residents::bytes_per_resident(),
        "history_record_payload_peak": history.peak * History::record_payload_bytes(),
        "history_record_payload_bound": HISTORY_CAPACITY * History::record_payload_bytes(),
        "anchor_frame_payload_bound": DEPTH * (4 * WIDTH * 4 + (TILES + 1) * 12 + 2 * WIDTH * 4 + 24),
        "returned_correction_payload_bound": DEPTH * WIDTH * 4,
        "pending_anchor_record_payload_bound": DEPTH * History::record_payload_bytes(),
        "shared_gradient_payload_bound": (2 * VOCAB * WIDTH + WIDTH) * 4,
        "predictor_gradient_payload": WIDTH * FEATURES * 4,
        "training_loss_payload_bound": options.steps.div_ceil(options.anchor_interval) * 4,
        "node_version_payload": TILES * 8,
        "single_node_gradient_payload_bound": (WIDTH * WIDTH + WIDTH + 7 * 2 * WIDTH) * 4,
        "excludes": "allocator/container overhead, stack, transient math copies, Rust/runtime/RSS; not a measured peak"
    })
}
