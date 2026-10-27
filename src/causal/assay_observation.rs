use super::assay_measure::Case;
use super::assay_source::Source;
use super::assay_state::{ObservationRef, Progress, GENERATIONS, MANIFEST_SHA, PASSES, PROTOCOL};
use super::checkpoint_owner::Terminal;
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

pub(super) fn read(directory: &Path, reference: &ObservationRef) -> Result<Vec<u8>> {
    let bytes = super::child_io::read(&directory.join(reference.name()), 65_536)?;
    ensure!(
        bytes.len() as u64 == reference.bytes
            && super::storage_paths::digest(&bytes) == reference.sha256,
        "assay observation reference changed"
    );
    let observation: Observation = serde_json::from_slice(&bytes)?;
    validate(&observation)?;
    ensure!(
        observation.schema == 1
            && observation.protocol == PROTOCOL
            && observation.manifest_sha256 == MANIFEST_SHA
            && observation.pass == reference.pass
            && observation.physical_anchors == reference.pass * 80
            && observation.numerical_clock == reference.pass * 320
            && observation.before_sha256 == observation.after_sha256
            && observation.complete == reference.complete
            && observation.teacher_readouts == reference.teacher_readouts
            && observation.generated_readouts == reference.generated_readouts,
        "assay observation contents changed"
    );
    Ok(bytes)
}

fn validate(value: &Observation) -> Result<()> {
    ensure!(
        super::storage_paths::hash_valid(&value.before_sha256)
            && value.cases.len() <= 48
            && PASSES.contains(&value.pass),
        "assay observation shape"
    );
    ensure!(
        matches!(
            value.interrupted,
            None | Some(Terminal::UserStop | Terminal::ResourceStop | Terminal::BudgetStop)
        ),
        "invalid assay interruption status"
    );
    let generate = GENERATIONS.contains(&value.pass);
    for (index, case) in value.cases.iter().enumerate() {
        let group = ["foundation", "paraphrase", "counterfactual"][index / 16];
        ensure!(
            case.id == format!("{group}-{:02}", index % 16)
                && case.content_nll.len() <= 4
                && case
                    .content_nll
                    .iter()
                    .chain(case.lf_nll.iter())
                    .all(|v| v.is_finite() && *v >= 0.0)
                && (case.lf_nll.is_none() || case.content_nll.len() == 4)
                && ((case.teacher_count() == 0 && case.first_byte_rank.is_none())
                    || (case.teacher_count() > 0
                        && case
                            .first_byte_rank
                            .is_some_and(|rank| (1..=256).contains(&rank)))),
            "assay case/teacher record invalid"
        );
        if let Some(raw) = &case.generated {
            ensure!(
                generate
                    && case.teacher_count() == 5
                    && raw.len() <= 8
                    && raw
                        .iter()
                        .position(|byte| *byte == b'\n')
                        .is_none_or(|position| position + 1 == raw.len()),
                "assay raw generation invalid"
            );
        }
        if case.generation_complete {
            let raw = case
                .generated
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing generation"))?;
            let word =
                ["soil", "sand", "clay", "silt"][(index % 16 + usize::from(index >= 32)) % 4];
            ensure!(
                !raw.is_empty()
                    && (raw.last() == Some(&b'\n') || raw.len() == 8)
                    && case.exact == Some(raw == format!("{word}\n").as_bytes()),
                "assay generation stop/exact invalid"
            );
        } else {
            ensure!(case.exact.is_none(), "censored exact-match claim");
        }
        ensure!(
            index + 1 == value.cases.len() || case.complete(generate),
            "nonterminal partial case"
        );
    }
    ensure!(
        value.teacher_readouts == value.cases.iter().map(Case::teacher_count).sum::<u64>()
            && value.generated_readouts
                == value.cases.iter().map(Case::generated_count).sum::<u64>()
            && value.complete
                == (value.interrupted.is_none()
                    && value.cases.len() == 48
                    && value.cases.iter().all(|case| case.complete(generate))),
        "assay coverage totals"
    );
    Ok(())
}

pub(super) fn verify_references(directory: &Path, state: &State) -> Result<()> {
    if let Some(progress) = &state.assay {
        for reference in &progress.observations {
            let value: Observation = serde_json::from_slice(&read(directory, reference)?)?;
            ensure!(
                value.seed == state.seed && value.rule == progress.rule,
                "assay observation seed/rule mismatch"
            );
        }
    }
    Ok(())
}

pub(super) fn copy_references(
    parent: &Path,
    destination: &Path,
    progress: &Progress,
) -> Result<()> {
    ensure!(
        progress.observations.iter().all(|entry| entry.complete),
        "censored observation cannot resume"
    );
    for reference in &progress.observations {
        let bytes = read(parent, reference)?;
        super::child_io::publish(destination, &reference.name(), &bytes)?;
    }
    Ok(())
}
