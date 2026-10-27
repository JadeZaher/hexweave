use super::child_protocol::{Operation, Request};
use super::control::{millis, Control};
use super::data::Dataset;
use super::evaluation::Coverage;
use super::prompt_evaluation::{self, Report};
use super::prompt_input::{self, Input};
use super::report::{EvaluationReport, Timings};
use super::runner_finish::{finish, save};
use super::state::State;
use super::storage::DiskStore;
use anyhow::{Context, Result};
use std::time::Instant;

pub(super) fn run(
    request: &Request,
    control: &mut Control<'_>,
    state: &State,
    store: &mut DiskStore,
    training: &Dataset,
) -> Result<serde_json::Value> {
    let Operation::PromptEvaluation { prompts } = &request.operation else {
        anyhow::bail!("prompt worker called for another operation");
    };
    let parent = request
        .resume
        .as_ref()
        .context("prompt evaluation has no parent")?;
    let bytes = prompt_input::read(&request.workspace, prompts)?;
    let input = Input::parse(
        &bytes,
        request.limits.generation_cases,
        request.limits.generation_bytes,
        state.geometry.context,
    )?;
    drop(bytes);
    let mut timings = Timings {
        initialization_millis: millis(control.started)?,
        ..Timings::default()
    };
    control.poll()?;
    let evaluation_checkpoint = save(store, state, training, control, 0, None, &mut timings)?;
    let started = Instant::now();
    let report = Report {
        input_sha256: prompts.sha256.clone(),
        parent_checkpoint: parent.checkpoint.clone(),
        evaluation_checkpoint,
        before_sha256: String::new(),
        after_sha256: String::new(),
        requested_cases: input.prompts.len(),
        cases: Vec::with_capacity(input.prompts.len()),
        coverage: Coverage::Complete,
    };
    let evaluated = prompt_evaluation::evaluate(
        state,
        store,
        &input,
        request.limits.generation_bytes,
        report,
        &mut || control.poll(),
    )?;
    timings.evaluation_millis = millis(started)?;
    drop(input);
    finish(
        request,
        control,
        state,
        store,
        training,
        0,
        EvaluationReport::PromptEvaluation(evaluated),
        timings,
    )
}
