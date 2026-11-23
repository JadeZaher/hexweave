//! Owned causal byte-model implementation; see AGENTS.md.

pub use decoding::Decoder;

mod assay_measure;
#[cfg(test)]
mod assay_observation;
mod assay_source;
mod assay_state;
mod assay_update;
mod attention;
mod attention_reverse;
mod checkpoint_state;
mod child_io;
mod config;
mod connected_adam;
mod connected_anchor;
mod connected_balance;
mod connected_calibration;
mod connected_checkpoint;
mod connected_checkpoint_io;
mod connected_completion;
mod connected_config;
mod connected_context;
mod connected_context_gate;
mod connected_decode;
mod connected_eval;
mod connected_eval_config;
mod connected_gain;
mod connected_lane_apply;
mod connected_lanes;
mod connected_plasticity;
mod connected_precision;
#[cfg(test)]
mod connected_precision_contracts;
mod connected_predictor;
#[cfg(test)]
mod connected_predictor_contracts;
mod connected_quantization;
mod connected_reach;
mod connected_replay;
mod connected_restart;
mod connected_restart_config;
mod connected_restart_due;
mod connected_restart_replay;
mod connected_route;
#[cfg(test)]
mod connected_route_contracts;
mod connected_sections;
mod connected_sections_read;
mod connected_sections_write;
mod connected_signal;
mod connected_simd;
mod connected_state;
mod connected_step;
mod connected_store;
mod connected_store_codec;
mod connected_store_read;
mod connected_tile_signal;
mod connected_train;
mod connected_update;
mod connected_update_scale;
mod connected_wave;
mod core_ahead;
mod core_batch;
mod core_best;
mod core_checkpoint;
#[cfg(test)]
mod core_codec;
mod core_completion;
mod core_config;
mod core_corpus;
mod core_drill;
mod core_eval;
mod core_gradients;
mod core_log;
mod core_redeal;
mod core_route;
mod core_schedule;
mod core_stream;
mod core_stream_read;
mod core_train;
mod credit;
mod cursor;
mod data;
mod data_contracts;
mod data_guard;
mod data_io;
mod data_provenance;
mod decoding;
mod evaluation;
mod execution;
#[cfg(test)]
mod execution_legacy;
mod ffn;
mod generation;
mod geometry;
mod history;
mod immutable;
mod norm;
mod positions;
mod predictor;
#[cfg(test)]
mod predictor_joint;
mod pulse;
mod replay;
mod resident;
mod routing;
mod shared;
mod state;
mod state_codec;
mod state_counters;
mod storage_paths;
mod store_api;
mod telemetry;
mod tile;
mod tile_gradients;
mod tuning;
mod tuning_measure;
mod tuning_sources;
mod tuning_update;
mod weights;

#[cfg(test)]
mod assay_contracts;
#[cfg(test)]
mod attention_contracts;
#[cfg(test)]
mod connected_bias_contracts;
#[cfg(test)]
mod connected_context_gradient_contracts;
#[cfg(test)]
mod connected_contracts;
#[cfg(test)]
mod connected_curve_contracts;
#[cfg(test)]
mod connected_forward_contracts;
#[cfg(test)]
mod connected_golden_contracts;
#[cfg(test)]
mod connected_numeric_contracts;
#[cfg(test)]
mod connected_paging_contracts;
#[cfg(test)]
mod connected_restart_contracts;
#[cfg(test)]
mod connected_restart_option_contracts;
#[cfg(test)]
mod connected_storage_contracts;
#[cfg(test)]
mod core_batch_contracts;
#[cfg(test)]
mod core_contracts;
#[cfg(test)]
mod core_paged_experiment;
#[cfg(test)]
mod core_redeal_contracts;
#[cfg(test)]
mod core_reproduction_contract;
#[cfg(test)]
mod core_resume_contracts;
#[cfg(test)]
mod data_schema_contracts;
#[cfg(test)]
mod data_tests;
#[cfg(test)]
mod execution_contracts;
#[cfg(test)]
mod float_contracts;
#[cfg(test)]
mod generation_contracts;
#[cfg(test)]
mod history_contracts;
#[cfg(test)]
mod numerical_tests;
#[cfg(test)]
mod parameter_contracts;
#[cfg(test)]
mod replay_contracts;
#[cfg(test)]
mod scratch_dir;
#[cfg(test)]
mod tuning_harness;

pub fn plan() -> anyhow::Result<serde_json::Value> {
    config::plan()
}

/// Core training path: `hexweave train --config <json>`; see AGENTS.md §core.
pub fn train(config: &std::path::Path) -> anyhow::Result<serde_json::Value> {
    core_train::run(config)
}

/// Generate from a verified corpus checkpoint; see AGENTS.md.
pub fn complete(
    config: &std::path::Path,
    checkpoint: &str,
    prompt: &str,
    limit: usize,
) -> anyhow::Result<serde_json::Value> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| core_completion::complete(config, checkpoint, prompt, limit))
            .join()
            .map_err(|_| anyhow::anyhow!("completion thread panicked"))?
    })
}

/// Held-out loss by retained context length for a connected checkpoint; see CONNECTED.md.
pub fn context_reach(
    config: &std::path::Path,
    checkpoint: &str,
    spans: usize,
) -> anyhow::Result<serde_json::Value> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| connected_completion::reach_path(config, checkpoint, spans))
            .join()
            .map_err(|_| anyhow::anyhow!("context reach thread panicked"))?
    })
}

/// Complete with a chosen decoder; connected checkpoints also take bounded decoder diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn complete_audit(
    config: &std::path::Path,
    checkpoint: &str,
    prompt: &str,
    limit: usize,
    trace_bytes: usize,
    generated_feedback: bool,
    quantize_audit: bool,
    decoder: Decoder,
) -> anyhow::Result<serde_json::Value> {
    let decode = connected_decode::Options {
        trace_bytes,
        generated_feedback,
        quantize_audit,
        decoder,
    };
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                core_completion::complete_options(
                    config, checkpoint, prompt, limit, decode, decoder,
                )
            })
            .join()
            .map_err(|_| anyhow::anyhow!("completion thread panicked"))?
    })
}
