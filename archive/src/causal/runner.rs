use super::checkpoint;
use super::checkpoint_owner::ResumeRequest;
use super::child_protocol::Request;
use super::control::{millis, Control};
use super::data::Dataset;
use super::data_contracts::Split;
use super::evaluation::{self, Evaluation};
use super::execution::{self, Step};
use super::geometry::Geometry;
use super::report::{EvaluationReport, Evaluations, Timings};
use super::runner_finish::{finish, save};
use super::state::State;
use super::storage::DiskStore;
use super::{child_io, launch, pinning};
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::time::Instant;

pub(super) fn owned(path: &Path) -> Result<serde_json::Value> {
    let started = Instant::now();
    let bytes = child_io::read(path, child_io::REQUEST_CAP)?;
    let request: Request = serde_json::from_slice(&bytes)?;
    let owner_process = launch::verify(path, &bytes, &request)?;
    request.validate()?;
    pinning::verify(&request)?;
    let mut control = Control::new(&request, owner_process, started);
    if request.config.assay_rule().is_some() {
        return super::assay_runner::run(&request, &mut control);
    }
    let training = Dataset::open(
        &request.data_root,
        &request.data_manifest_sha256,
        Split::Train,
    )?;
    ensure!(
        training.identity() == &request.identity.data,
        "worker data identity changed"
    );
    let (mut store, mut state) = initialize(&request, &training)?;
    if request.operation.is_prompt() {
        return super::runner_prompts::run(&request, &mut control, &state, &mut store, &training);
    }
    let development = Dataset::open(
        &request.data_root,
        &request.data_manifest_sha256,
        Split::Development,
    )?;
    let mut timings = Timings {
        initialization_millis: millis(started)?,
        ..Timings::default()
    };
    control.poll()?;
    save(
        &mut store,
        &state,
        &training,
        &control,
        0,
        None,
        &mut timings,
    )?;
    let initial = evaluate(&state, &mut store, &development, &mut control, &mut timings)?;
    let physical = train(
        &mut state,
        &mut store,
        &training,
        &mut control,
        &mut timings,
    )?;
    let evaluation_checkpoint = save(
        &mut store,
        &state,
        &training,
        &control,
        physical,
        None,
        &mut timings,
    )?;
    let final_evaluation = evaluate(&state, &mut store, &development, &mut control, &mut timings)?;
    finish(
        &request,
        &mut control,
        &state,
        &mut store,
        &training,
        physical,
        EvaluationReport::Train(Evaluations {
            initial,
            r#final: final_evaluation,
            omitted_reason: None,
            final_evaluation_checkpoint: evaluation_checkpoint,
        }),
        timings,
    )
}

fn initialize(request: &Request, training: &Dataset) -> Result<(DiskStore, State)> {
    if let Some(resume) = &request.resume {
        let parent = resume
            .parent_directory
            .parent()
            .context("missing parent attempt")?;
        let continuation = ResumeRequest {
            release_sha256: resume.release_sha256.clone(),
            release_seal_sha256: resume.release_seal_sha256.clone(),
            explicit_user_renewal: resume.explicit_user_renewal,
            new_owner: request.owner.clone(),
        };
        if request.operation.is_prompt() {
            let old = checkpoint::inspect(&resume.parent_directory, &resume.checkpoint)?;
            let release = super::checkpoint_release::read(
                &request.workspace,
                parent,
                &continuation,
                &request.identity,
                &old.owner,
            )?;
            ensure!(
                release.status == super::checkpoint_owner::Terminal::Completed
                    && old.owner.stop == super::checkpoint_owner::Stop::Running,
                "prompt evaluation requires clean Completed parent"
            );
        }
        let (store, state) = checkpoint::restore(
            &request.workspace,
            parent,
            &resume.checkpoint,
            &request.attempt_directory,
            &request.identity,
            &continuation,
            training,
        )?;
        ensure!(
            state.seed == request.config.seed,
            "restored seed/config mismatch"
        );
        Ok((store, state))
    } else {
        let state = State::new(Geometry::candidate(), request.config.seed, training)?;
        let store = DiskStore::create(
            &request.workspace,
            &request.attempt_directory,
            state.geometry,
            state.seed,
            request.owner.budget.disk_bytes_per_attempt,
        )?;
        Ok((store, state))
    }
}

fn evaluate(
    state: &State,
    store: &mut DiskStore,
    dataset: &Dataset,
    control: &mut Control<'_>,
    timings: &mut Timings,
) -> Result<Option<Evaluation>> {
    if control.poll()?.is_some() {
        return Ok(None);
    }
    let started = Instant::now();
    let limits = &control.request.limits;
    let limits = evaluation::Limits {
        targets: limits.evaluation_targets,
        cases: limits.generation_cases,
        bytes_per_case: limits.generation_bytes,
    };
    let result = evaluation::evaluate(state, store, dataset, limits, &mut || control.poll())?;
    timings.evaluation_millis = timings
        .evaluation_millis
        .checked_add(millis(started)?)
        .context("evaluation timing overflow")?;
    Ok(Some(result))
}

fn train(
    state: &mut State,
    store: &mut DiskStore,
    training: &Dataset,
    control: &mut Control<'_>,
    timings: &mut Timings,
) -> Result<u64> {
    let mut physical = 0u64;
    while physical < control.request.limits.opportunities {
        if control.poll()?.is_some() {
            break;
        }
        let started = Instant::now();
        let result = execution::step(state, store, training, false)?;
        ensure!(
            matches!(result, Step::Committed { .. }),
            "entered opportunity did not commit"
        );
        physical = physical
            .checked_add(1)
            .context("physical opportunity overflow")?;
        timings.training_millis = timings
            .training_millis
            .checked_add(millis(started)?)
            .context("training timing overflow")?;
        if physical.is_multiple_of(control.request.limits.checkpoint_every) {
            save(store, state, training, control, physical, None, timings)?;
        }
    }
    Ok(physical)
}
