use super::fixture::Store;
use super::identity::identity;
use super::source::{ANSWERS, HELD_OUT, TRAIN};
use crate::causal::execution::forward_path;
use crate::causal::generation::lowest_maximum;
use crate::causal::shared::Readout;
use crate::causal::state::State;
use anyhow::{ensure, Result};
use serde::Serialize;
use std::time::{Duration, Instant};

pub(super) struct Budget {
    started: Instant,
    pub teacher_readouts: usize,
    pub generated_readouts: usize,
}

impl Budget {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            teacher_readouts: 0,
            generated_readouts: 0,
        }
    }

    pub fn check(&self) -> Result<()> {
        ensure!(
            self.started.elapsed() < Duration::from_secs(300),
            "five-minute fixture abort ceiling reached"
        );
        Ok(())
    }
}

#[derive(Serialize)]
pub(super) struct Observation {
    pub completed_cycles: usize,
    pub before_fixture_sha256: String,
    pub after_fixture_sha256: String,
    pub train_nll: [f32; 8],
    pub held_out_nll: [f32; 8],
    pub train_mean_nll: f64,
    pub held_out_mean_nll: f64,
    pub train_digit_mean_nll: f64,
    pub greedy: Vec<Continuation>,
}

#[derive(Serialize)]
pub(super) struct Continuation {
    case_index: usize,
    raw_bytes: Vec<u8>,
    stop_reason: &'static str,
    distinct_bytes: usize,
    longest_repeated_run: usize,
    exact_two_byte_match: bool,
}

pub(super) fn observe(
    state: &State,
    store: &mut Store,
    cycles: usize,
    budget: &mut Budget,
) -> Result<Observation> {
    budget.check()?;
    let before = identity(state, store)?;
    let train = teacher(state, store, &TRAIN, budget)?;
    let held_out = teacher(state, store, &HELD_OUT, budget)?;
    let mut greedy = Vec::with_capacity(2);
    if cycles == 0 || cycles == 8 {
        for index in [0, 3] {
            greedy.push(generate(state, store, index, budget)?);
        }
    }
    let after = identity(state, store)?;
    ensure!(before == after, "fixture measurement mutated state");
    budget.check()?;
    Ok(Observation {
        completed_cycles: cycles,
        before_fixture_sha256: before,
        after_fixture_sha256: after,
        train_mean_nll: mean(&train),
        held_out_mean_nll: mean(&held_out),
        train_digit_mean_nll: mean(&train[..4]),
        train_nll: train,
        held_out_nll: held_out,
        greedy,
    })
}

fn mean(values: &[f32]) -> f64 {
    values.iter().map(|&x| f64::from(x)).sum::<f64>() / values.len() as f64
}

fn readout(state: &State, store: &mut Store, visible: &[u8]) -> Result<Readout> {
    ensure!((1..=65).contains(&visible.len()), "fixture prefix cap");
    let prefix = state.shared.prefix(visible)?;
    let trace = forward_path(
        store,
        state.geometry,
        &prefix,
        vec![0.0; state.geometry.width],
        &mut None,
        state.tile_versions,
    )?;
    state.shared.readout(&trace.output)
}

fn teacher(
    state: &State,
    store: &mut Store,
    prefixes: &[&[u8]; 4],
    budget: &mut Budget,
) -> Result<[f32; 8]> {
    let mut losses = [0.0; 8];
    for offset in 0..2 {
        for case in 0..4 {
            budget.check()?;
            ensure!(budget.teacher_readouts < 288, "teacher readout cap");
            let mut visible = prefixes[case].to_vec();
            visible.extend_from_slice(&ANSWERS[case][..offset]);
            let result = readout(state, store, &visible)?;
            losses[offset * 4 + case] = result.loss(ANSWERS[case][offset])?;
            budget.teacher_readouts += 1;
        }
    }
    ensure!(losses.iter().all(|x| x.is_finite()), "fixture NLL");
    Ok(losses)
}

fn generate(
    state: &State,
    store: &mut Store,
    case_index: usize,
    budget: &mut Budget,
) -> Result<Continuation> {
    let mut visible = HELD_OUT[case_index].to_vec();
    let mut raw_bytes = Vec::with_capacity(16);
    let mut stop_reason = "byte_budget";
    for _ in 0..16 {
        budget.check()?;
        ensure!(budget.generated_readouts < 384, "generated readout cap");
        let result = readout(state, store, &visible)?;
        let byte = u8::try_from(lowest_maximum(&result.logits)?)?;
        budget.generated_readouts += 1;
        raw_bytes.push(byte);
        visible.push(byte);
        if byte == b'\n' {
            stop_reason = "newline";
            break;
        }
    }
    let mut counts = [0usize; 256];
    let (mut run, mut longest) = (0, 0);
    for (index, &byte) in raw_bytes.iter().enumerate() {
        counts[usize::from(byte)] += 1;
        run = if index > 0 && raw_bytes[index - 1] == byte {
            run + 1
        } else {
            1
        };
        longest = longest.max(run);
    }
    Ok(Continuation {
        case_index,
        exact_two_byte_match: raw_bytes == ANSWERS[case_index],
        raw_bytes,
        stop_reason,
        distinct_bytes: counts.iter().filter(|&&count| count > 0).count(),
        longest_repeated_run: longest,
    })
}
