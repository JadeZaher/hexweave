//! Owned causal byte-model implementation; see AGENTS.md.

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
mod core_ahead;
mod core_best;
mod core_checkpoint;
mod core_codec;
mod core_config;
mod core_corpus;
mod core_drill;
mod core_eval;
mod core_log;
mod core_redeal;
mod core_train;
mod credit;
mod cursor;
mod data;
mod data_contracts;
mod data_guard;
mod data_io;
mod data_provenance;
mod evaluation;
mod execution;
mod ffn;
mod generation;
mod geometry;
mod history;
mod immutable;
mod norm;
mod positions;
mod predictor;
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
mod weights;

#[cfg(test)]
mod assay_contracts;
#[cfg(test)]
mod attention_contracts;
#[cfg(test)]
mod core_contracts;
#[cfg(test)]
mod core_redeal_contracts;
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
mod numerical_tests;
#[cfg(test)]
mod parameter_contracts;
#[cfg(test)]
mod replay_contracts;
#[cfg(test)]
mod tuning_harness;

pub fn plan() -> anyhow::Result<serde_json::Value> {
    config::plan()
}

/// Core training path: `hexweave train --config <json>`; see AGENTS.md §core.
pub fn train(config: &std::path::Path) -> anyhow::Result<serde_json::Value> {
    core_train::run(config)
}
