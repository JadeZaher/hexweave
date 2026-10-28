use super::assay_source::Source;
use super::assay_state::{Progress, ANCHORS, PASSES};
use super::child_protocol::Request;
use super::control::{millis, Control};
use super::report::{EvaluationReport, Timings};
use super::runner_finish::{finish, save};
use super::state::State;
use super::storage::DiskStore;
use anyhow::{ensure, Context, Result};
use std::time::Instant;

fn initialize(request: &Request, source: &Source) -> Result<(DiskStore, State)> {
    let rule = request.config.assay_rule().context("missing assay rule")?;
    if let Some(resume) = &request.resume {
        let parent = resume
            .parent_directory
            .parent()
            .context("missing parent attempt")?;
        let continuation = super::checkpoint_owner::ResumeRequest {
            release_sha256: resume.release_sha256.clone(),
            release_seal_sha256: resume.release_seal_sha256.clone(),
            explicit_user_renewal: resume.explicit_user_renewal,
            new_owner: request.owner.clone(),
        };
        let (store, state) = super::checkpoint::restore(
            &request.workspace,
            parent,
            &resume.checkpoint,
            &request.attempt_directory,
            &request.identity,
            &continuation,
            source,
        )?;
        ensure!(
            state.seed == request.config.seed
                && state.assay.as_ref().is_some_and(|a| a.rule == rule),
            "restored assay seed/rule mismatch"
        );
        Ok((store, state))
    } else {
        let mut state = State::with_cursor(
            super::geometry::Geometry::candidate(),
            request.config.seed,
            super::assay_source::cursor(0)?,
        )?;
        state.assay = Some(Progress::new(rule));
        state.validate_boundary()?;
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

pub(super) fn run(request: &Request, control: &mut Control<'_>) -> Result<serde_json::Value> {
    let source = Source::open(&request.data_root, &request.data_manifest_sha256)?;
    let (mut store, mut state) = initialize(request, &source)?;
    let mut timings = Timings {
        initialization_millis: millis(control.started)?,
        ..Timings::default()
    };
    let start = state.global_opportunity;
    ensure!(
        start == request.owner.attempt_start_opportunity,
        "assay physical resume offset"
    );
    save(&mut store, &state, &source, control, 0, None, &mut timings)?;
    loop {
        if control.poll()?.is_some() {
            break;
        }
        if !observe_due(&mut state, &mut store, &source, control, &mut timings)? {
            break;
        }
        if state.global_opportunity == ANCHORS {
            break;
        }
        if control.poll()?.is_some() {
            break;
        }
        let began = Instant::now();
        ensure!(
            matches!(
                super::execution::step(&mut state, &mut store, &source, false)?,
                super::execution::Step::Committed { .. }
            ),
            "assay opportunity did not commit"
        );
        timings.training_millis += millis(began)?;
        let physical = state.global_opportunity - start;
        if physical.is_multiple_of(request.limits.checkpoint_every) {
            save(
                &mut store,
                &state,
                &source,
                control,
                physical,
                None,
                &mut timings,
            )?;
        }
    }
    Source::open(&request.data_root, &request.data_manifest_sha256)?;
    let progress = state.assay.clone().context("assay state absent")?;
    for reference in &progress.observations {
        super::assay_observation::read(&request.attempt_directory, reference)?;
    }
    finish(
        request,
        control,
        &state,
        &mut store,
        &source,
        state.global_opportunity - start,
        EvaluationReport::RetainedAssay(progress),
        timings,
    )
}

fn observe_due(
    state: &mut State,
    store: &mut DiskStore,
    source: &Source,
    control: &mut Control<'_>,
    timings: &mut Timings,
) -> Result<bool> {
    let next = state
        .assay
        .as_ref()
        .context("assay state absent")?
        .observations
        .len();
    if PASSES
        .get(next)
        .is_none_or(|pass| pass * 80 != state.global_opportunity)
    {
        return Ok(true);
    }
    let began = Instant::now();
    let observation =
        super::assay_observation::observe(state, store, source, &mut || control.poll())?;
    let reference =
        super::assay_observation::publish(&control.request.attempt_directory, &observation)?;
    state
        .assay
        .as_mut()
        .context("assay absent")?
        .observations
        .push(reference);
    timings.evaluation_millis += millis(began)?;
    save(
        store,
        state,
        source,
        control,
        state.global_opportunity - control.request.owner.attempt_start_opportunity,
        None,
        timings,
    )?;
    Ok(observation.complete)
}
