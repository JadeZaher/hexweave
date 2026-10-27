use super::checkpoint_owner::{OwnerEnvelope, Terminal};
use super::checkpoint_state::{CheckpointRef, Identity};
use super::child_protocol::Operation;
use super::cursor::Cursor;
use super::evaluation::Evaluation;
use super::state::Counters;
use super::storage::IoCounters;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Timings {
    pub initialization_millis: u64,
    pub training_millis: u64,
    pub evaluation_millis: u64,
    pub checkpoint_millis: u64,
    pub total_child_millis: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Evaluations {
    pub initial: Option<Evaluation>,
    pub r#final: Option<Evaluation>,
    pub omitted_reason: Option<Terminal>,
    pub final_evaluation_checkpoint: CheckpointRef,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "result")]
#[allow(clippy::large_enum_variant)]
pub(super) enum EvaluationReport {
    Train(Evaluations),
    RetainedAssay(super::assay_state::Progress),
    PromptEvaluation(super::prompt_evaluation::Report),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerResult {
    pub schema: u32,
    pub protocol: String,
    pub status: Terminal,
    pub attempt_id: String,
    pub checkpoint: CheckpointRef,
    pub global_opportunity: u64,
    pub physical_opportunities_completed: u64,
    pub identity: Identity,
    pub owner: OwnerEnvelope,
    pub cursor: Cursor,
    pub counters: Counters,
    pub io: IoCounters,
    pub operation: Operation,
    pub evaluation: EvaluationReport,
    pub timings: Timings,
    pub training_admission_claim: bool,
    pub useful_assistant_claim: bool,
}
