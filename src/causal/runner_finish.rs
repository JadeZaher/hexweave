use super::checkpoint::publish;
use super::checkpoint_state::CheckpointRef;
use super::child_protocol::Request;
use super::control::{millis, Control};
use super::execution::TrainingSource;
use super::report::{EvaluationReport, Timings, WorkerResult};
use super::state::State;
use super::storage::DiskStore;
use super::{checkpoint_retention, child_io, pinning};
use anyhow::{Context, Result};
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub(super) fn save<S: TrainingSource>(
    store: &mut DiskStore,
    state: &State,
    training: &S,
    control: &Control<'_>,
    physical: u64,
    terminal: Option<super::checkpoint_owner::Terminal>,
    timings: &mut Timings,
) -> Result<CheckpointRef> {
    let started = Instant::now();
    let owner = control.envelope(physical, terminal)?;
    let checkpoint = publish(store, state, &control.request.identity, &owner, training)?;
    checkpoint_retention::retain(store, checkpoint_retention::RECEIPT_GENERATIONS)?;
    timings.checkpoint_millis = timings
        .checkpoint_millis
        .checked_add(millis(started)?)
        .context("checkpoint timing overflow")?;
    Ok(checkpoint)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finish<S: TrainingSource>(
    request: &Request,
    control: &mut Control<'_>,
    state: &State,
    store: &mut DiskStore,
    training: &S,
    physical: u64,
    mut evaluations: EvaluationReport,
    mut timings: Timings,
) -> Result<serde_json::Value> {
    pinning::verify(request)?;
    let mut status = control.terminal(physical)?;
    if let EvaluationReport::RetainedAssay(report) = &evaluations {
        anyhow::ensure!(
            status != super::checkpoint_owner::Terminal::Completed
                || (state.global_opportunity == 1600 && report.complete()),
            "incomplete assay cannot finish Completed"
        );
    }
    if let EvaluationReport::Train(report) = &mut evaluations {
        if report.initial.is_none() || report.r#final.is_none() {
            report.omitted_reason = Some(status);
        }
    }
    if let super::child_protocol::Operation::PromptEvaluation { prompts } = &request.operation {
        super::prompt_input::read(&request.workspace, prompts)?;
    }
    let mut checkpoint = save(
        store,
        state,
        training,
        control,
        physical,
        Some(status),
        &mut timings,
    )?;
    let after_checkpoint = control.terminal(physical)?;
    if after_checkpoint != status {
        status = after_checkpoint;
        checkpoint = save(
            store,
            state,
            training,
            control,
            physical,
            Some(status),
            &mut timings,
        )?;
    }
    timings.total_child_millis = millis(control.started)?;
    let result = WorkerResult {
        schema: 1,
        protocol: request.protocol.clone(),
        status,
        attempt_id: request.owner.attempt_id.clone(),
        checkpoint,
        global_opportunity: state.global_opportunity,
        physical_opportunities_completed: physical,
        identity: request.identity.clone(),
        owner: control.envelope(physical, Some(status))?,
        cursor: state.cursor.clone(),
        counters: state.counters.clone(),
        io: store.io.clone(),
        operation: request.operation.clone(),
        evaluation: evaluations,
        timings,
        training_admission_claim: false,
        useful_assistant_claim: false,
    };
    let bytes = child_io::json(&result)?;
    let sha256 = child_io::publish(&request.attempt_directory, "result.json", &bytes)?;
    Ok(
        serde_json::json!({"schema":1,"protocol":request.protocol,"attempt_id":request.owner.attempt_id,
        "result_sha256":sha256,"status":status,"result_file":"result.json"}),
    )
}
