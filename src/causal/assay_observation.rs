use super::assay_measure::Case;
use super::assay_source::Source;
use super::assay_state::{ObservationRef, GENERATIONS, MANIFEST_SHA, PASSES, PROTOCOL};
use super::generation::Terminal;
use super::state::State;
use super::store_api::TileStore;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Observation {
    pub schema: u32,
    pub protocol: String,
    pub manifest_sha256: String,
    pub seed: u64,
    pub rule: super::assay_update::Rule,
    pub pass: u64,
    pub physical_anchors: u64,
    pub numerical_clock: u64,
    pub before_sha256: String,
    pub after_sha256: String,
    pub complete: bool,
    pub interrupted: Option<Terminal>,
    pub teacher_readouts: u64,
    pub generated_readouts: u64,
    pub cases: Vec<Case>,
}

pub(super) fn observe<S: TileStore, F: FnMut() -> Result<Option<Terminal>>>(
    state: &State,
    store: &mut S,
    source: &Source,
    poll: &mut F,
) -> Result<Observation> {
    let progress = state
        .assay
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing assay state"))?;
    let pass = state.global_opportunity / 80;
    ensure!(
        state.global_opportunity.is_multiple_of(80)
            && PASSES.get(progress.observations.len()) == Some(&pass),
        "unexpected observation boundary"
    );
    let before_sha256 = super::immutable::identity(state, store)?;
    let generate = GENERATIONS.contains(&pass);
    let mut cases = Vec::with_capacity(48);
    let mut interrupted = None;
    for row in source.observations()? {
        if let Some(stop) = poll()? {
            interrupted = Some(stop);
            break;
        }
        let case = super::assay_measure::measure(state, store, row, generate, &mut || {
            let stop = poll()?;
            if stop.is_some() {
                interrupted = stop;
            }
            Ok(stop)
        })?;
        let complete = case.complete(generate);
        cases.push(case);
        if !complete {
            break;
        }
    }
    if let Some(stop) = poll()? {
        interrupted = Some(stop);
    }
    let after_sha256 = super::immutable::identity(state, store)?;
    ensure!(
        before_sha256 == after_sha256,
        "assay observation mutated numerical identity"
    );
    let complete = interrupted.is_none()
        && cases.len() == 48
        && cases.iter().all(|case| case.complete(generate));
    Ok(Observation {
        schema: 1,
        protocol: PROTOCOL.into(),
        manifest_sha256: MANIFEST_SHA.into(),
        seed: state.seed,
        rule: progress.rule,
        pass,
        physical_anchors: state.global_opportunity,
        numerical_clock: state.numerical_clock()?,
        before_sha256,
        after_sha256,
        complete,
        interrupted,
        teacher_readouts: cases.iter().map(Case::teacher_count).sum(),
        generated_readouts: cases.iter().map(Case::generated_count).sum(),
        cases,
    })
}

pub(super) fn publish(directory: &Path, observation: &Observation) -> Result<ObservationRef> {
    let bytes = super::child_io::json(observation)?;
    ensure!(bytes.len() <= 65_536, "assay observation byte bound");
    let reference = ObservationRef {
        pass: observation.pass,
        sha256: super::storage_paths::digest(&bytes),
        bytes: bytes.len() as u64,
        complete: observation.complete,
        teacher_readouts: observation.teacher_readouts,
        generated_readouts: observation.generated_readouts,
    };
    super::child_io::publish(directory, &reference.name(), &bytes)?;
    Ok(reference)
}
