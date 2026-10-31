//! Core training path (`hexweave train --config`): latent Adam, greedy routes, anchor-only,
//! resident model, hash-manifested resumable checkpoints. See AGENTS.md §core.

use super::assay_source::Source as AssaySource;
use super::assay_state::MANIFEST_SHA;
use super::core_ahead::Ahead;
use super::core_best;
use super::core_checkpoint::{self, PROTOCOL};
use super::core_config::{Config, Data};
use super::core_corpus::Corpus;
use super::core_drill::Drill;
use super::core_eval;
use super::core_log::{self, Evaluate, Marks, RecordState, Run};
use super::core_redeal::CitationRedeal;
use super::data::Dataset;
use super::data_contracts::Split;
use super::execution::{self, TrainingSource};
use super::geometry::Geometry;
use super::resident::ResidentStore;
use super::state::State;
use super::telemetry::{self, Phase};
use super::tuning;
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

/// Run (or resume) one core training job from a JSON config.
pub(super) fn run(path: &Path) -> Result<Value> {
    let config: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    config.validate()?;
    std::fs::create_dir_all(&config.out)?;
    tuning::install(config.knobs());
    let geometry = Geometry::candidate();
    match &config.data {
        Data::Science {
            root,
            manifest_sha256,
            redeal_citations,
        } => {
            let train = Dataset::open(root, manifest_sha256, Split::Train)?;
            let dev = Dataset::open(root, manifest_sha256, Split::Development)?;
            let start = State::new(geometry, config.seed, &train)?;
            let cases = config.eval_cases;
            let evaluate = &mut |state: &State, store: &mut ResidentStore| {
                core_eval::science(state, store, &train, &dev, cases)
            };
            if *redeal_citations {
                let source = CitationRedeal {
                    inner: &train,
                    seed: config.seed,
                };
                drive(&config, &source, start, evaluate, false, false)
            } else {
                drive(&config, &train, start, evaluate, false, false)
            }
        }
        Data::Drill { root } => {
            let assay = AssaySource::open(root, MANIFEST_SHA)?;
            let drill = Drill::new(&assay, config.seed)?;
            let start = State::with_cursor(geometry, config.seed, drill.cursor(0)?)?;
            drive(
                &config,
                &drill,
                start,
                &mut |state, store| core_eval::drill(state, store, &assay),
                false,
                false,
            )
        }
        Data::Corpus {
            root,
            manifest_sha256,
            window,
        } => {
            let corpus = Corpus::open(root, manifest_sha256, *window)?;
            let short = corpus.is_short();
            let start = State::with_cursor(geometry, config.seed, corpus.cursor(0)?)?;
            let eval_spans = config.eval_spans;
            drive(
                &config,
                &corpus,
                start,
                &mut |state, store| core_eval::corpus(state, store, &corpus, eval_spans),
                true,
                short,
            )
        }
    }
}

/// Newest verifying checkpoint when resuming (none: fresh); a fresh run refuses existing checkpoints.
fn open(config: &Config, mut start: State, identity: &str) -> Result<(State, ResidentStore)> {
    let geometry = start.geometry;
    if config.resume {
        if let Some((state, store, _)) = core_checkpoint::resume(&config.out, geometry, identity)? {
            return Ok((state, store));
        }
    } else if let Some(path) = core_checkpoint::latest(&config.out)? {
        bail!(
            "{} holds checkpoints; set resume to continue",
            path.display()
        );
    } else {
        core_best::ensure_fresh(&config.out)?;
    }
    start.anchor_only = true;
    start.validate_boundary()?;
    Ok((start, ResidentStore::new(geometry, config.seed)?))
}

/// Train one corpus run's cursor forward; `corpus` gates the train-ahead feed, `short` nulls it on a corpus shorter than `AHEAD_ANCHORS`.
fn drive<S: TrainingSource>(
    config: &Config,
    source: &S,
    start: State,
    evaluate: &mut Evaluate<'_>,
    corpus: bool,
    short: bool,
) -> Result<Value> {
    let started = Instant::now();
    ensure!(
        tuning::current() == config.knobs(),
        "core numerics profile is not installed"
    );
    let identity = config.identity()?;
    let run = Run {
        config,
        source,
        identity: &identity,
    };
    let (mut state, mut store) = open(config, start, &identity)?;
    source.validate(&state.cursor)?;
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.out.join("log.jsonl"))?;
    let (stop, deadline) = (
        config.out.join("STOP"),
        started
            .checked_add(Duration::from_secs(config.wall_seconds))
            .context("wall_seconds overflows the clock")?,
    );
    let mut marks = Marks::new(&state);
    let mut ahead = Ahead::arm(
        state.global_opportunity,
        state.global_opportunity != 0,
        short,
    );
    let mut best = core_best::best_so_far(&config.out, &identity, config.eval_spans);
    let mut last = if state.global_opportunity == 0 {
        Some(core_log::record(
            &mut RecordState {
                log: &mut log,
                marks: &mut marks,
                ahead: &mut ahead,
            },
            &state,
            &mut store,
            evaluate,
            started,
        )?)
    } else {
        None
    };
    let terminal = loop {
        if state.global_opportunity >= config.anchors {
            break "completed";
        }
        if stop.exists() {
            break "stopped";
        }
        if Instant::now() >= deadline {
            break "deadline";
        }
        let step = execution::step(&mut state, &mut store, source, false)?;
        core_log::feed_ahead(&mut ahead, corpus, &step);
        let n = state.global_opportunity;
        if n.is_multiple_of(config.checkpoint_every) {
            telemetry::time(Phase::Checkpoint, || {
                core_log::checkpoint(&run, &state, &store)
            })?;
        }
        last = if n.is_multiple_of(config.eval_every) {
            Some(core_log::record(
                &mut RecordState {
                    log: &mut log,
                    marks: &mut marks,
                    ahead: &mut ahead,
                },
                &state,
                &mut store,
                evaluate,
                started,
            )?)
        } else {
            None
        };
        if let Some(line) = &last {
            telemetry::time(Phase::Checkpoint, || {
                core_best::keep(
                    &config.out,
                    &state,
                    &store,
                    &identity,
                    line,
                    &mut best,
                    config.eval_spans,
                )
            });
        }
    };
    let (checkpoint, last) = core_log::finalize_last(
        &run,
        &state,
        &mut store,
        &mut RecordState {
            log: &mut log,
            marks: &mut marks,
            ahead: &mut ahead,
        },
        evaluate,
        started,
        last,
    )?;
    let n = state.global_opportunity;
    core_best::keep(
        &config.out,
        &state,
        &store,
        &identity,
        &last,
        &mut best,
        config.eval_spans,
    );
    let report = json!({
        "protocol": PROTOCOL, "terminal": terminal, "anchors": n, "epoch": state.cursor.epoch,
        "wall_seconds": started.elapsed().as_secs_f64(), "identity_sha256": identity,
        "checkpoint": checkpoint, "final": last,
    });
    std::fs::write(
        config.out.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}
