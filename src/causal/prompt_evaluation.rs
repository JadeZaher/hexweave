use super::checkpoint_owner::Terminal;
use super::checkpoint_state::CheckpointRef;
use super::evaluation::Coverage;
use super::generation::{self, Generation, Reason};
use super::prompt_input::Input;
use super::state::State;
use super::store_api::TileStore;
use super::{immutable, storage_paths};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub id: String,
    pub prompt_sha256: String,
    pub generation: Generation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Report {
    pub input_sha256: String,
    pub parent_checkpoint: CheckpointRef,
    pub evaluation_checkpoint: CheckpointRef,
    pub before_sha256: String,
    pub after_sha256: String,
    pub requested_cases: usize,
    pub cases: Vec<Case>,
    pub coverage: Coverage,
}

pub(super) fn evaluate<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    input: &Input,
    new_bytes: usize,
    mut report: Report,
    poll: &mut F,
) -> Result<Report> {
    report.before_sha256 = immutable::identity(state, store)?;
    for row in &input.prompts {
        if let Some(status) = poll()? {
            report.coverage = Coverage::Interrupted(status);
            break;
        }
        let generation =
            generation::generate(state, store, row.prompt.as_bytes(), new_bytes, poll)?;
        let interrupted = match generation.stop_reason {
            Reason::Interrupted(status) => Some(status),
            _ => None,
        };
        report.cases.push(Case {
            id: row.id.clone(),
            prompt_sha256: storage_paths::digest(row.prompt.as_bytes()),
            generation,
        });
        if let Some(status) = interrupted {
            report.coverage = Coverage::Interrupted(status);
            break;
        }
    }
    report.after_sha256 = immutable::identity(state, store)?;
    ensure!(
        report.before_sha256 == report.after_sha256,
        "prompt evaluation mutated scientific state"
    );
    Ok(report)
}
