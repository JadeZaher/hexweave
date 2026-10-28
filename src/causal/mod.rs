//! Owned causal byte-model implementation; see AGENTS.md.

mod assay_measure;
mod assay_observation;
mod assay_runner;
mod assay_source;
mod assay_state;
mod assay_study;
mod assay_update;
mod attention;
mod attention_reverse;
mod checkpoint;
mod checkpoint_owner;
mod checkpoint_release;
mod checkpoint_restore;
mod checkpoint_retention;
mod checkpoint_state;
mod child_io;
mod child_protocol;
mod config;
mod control;
mod core_checkpoint;
mod core_codec;
mod core_drill;
mod core_eval;
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
mod launch;
mod norm;
mod pinning;
mod positions;
mod predictor;
mod predictor_joint;
mod profile;
mod prompt_evaluation;
mod prompt_input;
mod pulse;
mod report;
mod resident;
mod routing;
mod runner;
mod runner_finish;
mod runner_prompts;
mod shared;
mod state;
mod state_codec;
mod state_counters;
mod storage;
mod storage_paths;
mod store_api;
mod tile;
mod tile_gradients;
mod tuning;
mod tuning_measure;
mod tuning_sources;

#[cfg(test)]
mod anchor_comparison_contracts;
#[cfg(test)]
mod assay_contracts;
#[cfg(test)]
mod attention_contracts;
#[cfg(test)]
mod core_contracts;
#[cfg(test)]
mod data_schema_contracts;
#[cfg(test)]
mod data_tests;
#[cfg(test)]
mod execution_contracts;
#[cfg(test)]
mod generation_contracts;
#[cfg(test)]
mod learning_gap_contracts;
#[cfg(test)]
mod numerical_tests;
#[cfg(test)]
mod parameter_contracts;
#[cfg(test)]
mod persistence_profile;
#[cfg(test)]
mod persistence_tests;
#[cfg(test)]
mod predictor_commit_contracts;
#[cfg(test)]
mod predictor_update_contracts;
#[cfg(test)]
mod prompt_contracts;
#[cfg(test)]
mod release_seal_contracts;
#[cfg(test)]
mod revision_contracts;
#[cfg(test)]
mod runner_contracts;
#[cfg(test)]
mod tuning_harness;

pub fn plan() -> anyhow::Result<serde_json::Value> {
    config::plan()
}

pub fn run_owned(request: &std::path::Path) -> anyhow::Result<serde_json::Value> {
    runner::owned(request)
}

/// Core training path: `hexweave train --config <json>`; see AGENTS.md §core.
pub fn train(config: &std::path::Path) -> anyhow::Result<serde_json::Value> {
    core_train::run(config)
}
