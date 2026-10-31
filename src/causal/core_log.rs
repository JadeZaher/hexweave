//! Core-path log line, checkpoint-and-finalize and the train-ahead feed; see AGENTS.md §core.

use super::core_ahead::Ahead;
use super::core_checkpoint;
use super::core_config::Config;
use super::execution::{self, TrainingSource};
use super::resident::ResidentStore;
use super::state::State;
use super::telemetry::{self, Phase};
use super::tuning;
use anyhow::Result;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// One evaluation closure: runs the data-specific probes and returns their JSON report.
pub(super) type Evaluate<'a> = dyn FnMut(&State, &mut ResidentStore) -> Result<Value> + 'a;

/// The fixed run inputs `checkpoint` and `finalize_last` share with `drive`.
pub(super) struct Run<'a, S> {
    pub config: &'a Config,
    pub source: &'a S,
    pub identity: &'a str,
}

/// Interval bookkeeping between log lines.
pub(super) struct Marks {
    pub loss: f64,
    pub anchors: u64,
    pub at: Instant,
}

impl Marks {
    pub fn new(state: &State) -> Self {
        Self {
            loss: state.counters.anchor_loss_sum,
            anchors: state.counters.anchors,
            at: Instant::now(),
        }
    }
}

/// `record`'s mutable recording state, also threaded through `finalize_last`.
pub(super) struct RecordState<'a> {
    pub log: &'a mut std::fs::File,
    pub marks: &'a mut Marks,
    pub ahead: &'a mut Ahead,
}

/// Per-tile max latent Adam step (index 0..TILES) plus tied E's step (index TILES).
fn latent_steps() -> Vec<u64> {
    let mut steps = vec![0u64; super::geometry::TILES + 1];
    for ((owner, _), step) in tuning::steps() {
        steps[owner] = steps[owner].max(step);
    }
    steps
}

/// Feed a just-committed anchor's loss into the train-ahead accumulator (corpus runs only).
pub(super) fn feed_ahead(ahead: &mut Ahead, corpus: bool, step: &execution::Step) {
    if let (
        true,
        execution::Step::Committed {
            loss: Some(loss), ..
        },
    ) = (corpus, step)
    {
        ahead.add(*loss);
    }
}

/// Save only a Ready state whose cursor the source accepts.
pub(super) fn checkpoint<S: TrainingSource>(
    run: &Run<'_, S>,
    state: &State,
    store: &ResidentStore,
) -> Result<PathBuf> {
    state.validate_boundary()?;
    run.source.validate(&state.cursor)?;
    core_checkpoint::save(&run.config.out, state, store, run.identity)
}

/// Write one evaluation/telemetry log line, pairing the train-ahead window with its start anchor.
pub(super) fn record(
    recording: &mut RecordState<'_>,
    state: &State,
    store: &mut ResidentStore,
    evaluate: &mut Evaluate<'_>,
    started: Instant,
) -> Result<Value> {
    let trained = state.counters.anchors - recording.marks.anchors;
    let seconds = recording.marks.at.elapsed().as_secs_f64();
    let evaluation = telemetry::time(Phase::Evaluate, || evaluate(state, store))?;
    let (train_ahead, window_start) = recording.ahead.take();
    let line = json!({
        "anchors": state.global_opportunity, "epoch": state.cursor.epoch,
        "train_loss": (state.counters.anchor_loss_sum - recording.marks.loss) / trained.max(1) as f64,
        "anchors_per_second": trained as f64 / seconds.max(1e-9),
        "elapsed_seconds": started.elapsed().as_secs_f64(), "eval": evaluation,
        "telemetry": telemetry::drain(seconds), "latent_steps": latent_steps(),
        "train_ahead_prequential_bits_per_byte": train_ahead,
        "train_ahead_window_start": window_start,
    });
    writeln!(recording.log, "{line}")?;
    recording.log.flush()?;
    *recording.marks = Marks::new(state);
    *recording.ahead = Ahead::arm(state.global_opportunity, false, recording.ahead.short());
    Ok(line)
}

/// Ensure the final anchors are checkpointed and logged, saving one more record if needed.
pub(super) fn finalize_last<S: TrainingSource>(
    run: &Run<'_, S>,
    state: &State,
    store: &mut ResidentStore,
    recording: &mut RecordState<'_>,
    evaluate: &mut Evaluate<'_>,
    started: Instant,
    last: Option<Value>,
) -> Result<(PathBuf, Value)> {
    let n = state.global_opportunity;
    let path = match core_checkpoint::latest(&run.config.out)? {
        Some(path) if path.ends_with(format!("a{n:012}")) => path,
        _ => telemetry::time(Phase::Checkpoint, || checkpoint(run, state, store))?,
    };
    let last = match last {
        Some(line) => line,
        None => record(recording, state, store, evaluate, started)?,
    };
    Ok((path, last))
}
