//! Owned fresh or resumable connected corpus training; see AGENTS.md.

use super::connected_checkpoint;
use super::connected_config::Options;
use super::connected_state::State;
use super::connected_store::Store;
use super::core_config::{Config, Data};
use super::core_corpus::Corpus;
use super::execution::TrainingSource;
use super::geometry::{Geometry, VOCAB};
use super::tuning;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[path = "connected_log.rs"]
mod log;

/// Provenance of a fresh run's readout bias, named in the report and its first log row.
const READOUT_PRIOR: &str = "corpus-log-unigram";

pub(super) fn run(config: &Config) -> Result<Value> {
    config.validate()?;
    let options = config.connected.context("connected options missing")?;
    tuning::install(config.knobs());
    let Data::Corpus {
        root,
        manifest_sha256,
        window,
    } = &config.data
    else {
        anyhow::bail!("connected training requires a corpus");
    };
    let started = Instant::now();
    let corpus = Corpus::open(root, manifest_sha256, *window)?;
    std::fs::create_dir_all(&config.out)?;
    let identity = config.identity()?;
    let (mut state, mut store) = open(config, &corpus, options, &identity)?;
    store.set_paging(options.resident(), options.write_behind_tiles)?;
    corpus.validate(&state.cursor)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.out.join("log.jsonl"))?;
    let prior = (!config.resume).then_some(READOUT_PRIOR);
    let mut log = log::Log::new(file, &state, started, prior);
    let initial_anchors = state.anchors;
    let mut training_seconds = 0.0;
    let initial = evaluate(config, &state, &mut store, &corpus, &mut log, false)?;
    log.record(&state, &store, Some(initial), training_seconds)?;
    let terminal = drive(
        config,
        &mut state,
        &mut store,
        &corpus,
        &identity,
        &mut log,
        started,
        &mut training_seconds,
    )?;
    let path = checkpoint(config, &state, &mut store, &identity)?;
    let final_eval = evaluate(config, &state, &mut store, &corpus, &mut log, false)?;
    log.record(&state, &store, Some(final_eval.clone()), training_seconds)?;
    let mut report = json!({
        "schema": 1, "protocol": options.precision.protocol(), "precision": options.precision, "terminal": terminal, "anchors": state.anchors,
        "local_updates": state.counters.local_updates, "checkpoint": path,
        "elapsed_seconds": started.elapsed().as_secs_f64(), "training_seconds": training_seconds,
        "anchors_per_training_second": (state.anchors - initial_anchors) as f64 / training_seconds.max(1e-9),
        "eval": final_eval, "mechanism": state.counters, "paging": store.stats(),
        "gains": log::gains(&state, &store), "trust": log::trust(&state), "run_identity": identity,
        "from_scratch": !config.resume, "initial_anchors": initial_anchors, "readout_prior": READOUT_PRIOR,
        "memory_records": state.memory.records.len(), "predictor_fits": state.predictor.fits,
    });
    log::add_waves(&mut report, &state, &Default::default());
    if options.replay.is_some() {
        report["replay"] = state.replay.report();
    }
    if let Some(restart) = super::connected_restart::report(&state)? {
        report["restart"] = restart;
    }
    // Lifetime means in the report; log rows carry interval means.
    log::add_scale(&mut report, &state, &Default::default(), state.anchors);
    super::core_checkpoint::publish(
        &config.out.join("report.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn drive(
    config: &Config,
    state: &mut State,
    store: &mut Store,
    corpus: &Corpus,
    identity: &str,
    log: &mut log::Log,
    started: Instant,
    training_seconds: &mut f64,
) -> Result<&'static str> {
    let options = state.options;
    let mut last_heartbeat = Instant::now();
    let terminal = loop {
        if state.anchors >= config.anchors {
            break "completed";
        }
        if config.out.join("STOP").exists() {
            break "stopped";
        }
        if started.elapsed() >= Duration::from_secs(config.wall_seconds) {
            break "deadline";
        }
        let (at, before) = (Instant::now(), state.anchors);
        super::connected_wave::step(state, store, corpus, options, config.rate, true)?;
        *training_seconds += at.elapsed().as_secs_f64();
        if state.anchors.is_multiple_of(config.checkpoint_every) {
            checkpoint(config, state, store, identity)?;
        }
        let eval = if state.anchors.is_multiple_of(config.eval_every) {
            Some(evaluate(config, state, store, corpus, log, false)?)
        } else if (before + 1..=state.anchors).any(|anchors| {
            // A wave may pass a retention point; it is then probed at the wave's end.
            super::connected_eval::retention_due(anchors, config.connected_eval.unwrap_or_default())
        }) {
            Some(evaluate(config, state, store, corpus, log, true)?)
        } else {
            None
        };
        if eval.is_some() || last_heartbeat.elapsed() >= Duration::from_secs(5) {
            log.record(state, store, eval, *training_seconds)?;
            last_heartbeat = Instant::now();
        }
    };
    Ok(terminal)
}

fn open(
    config: &Config,
    corpus: &Corpus,
    options: Options,
    identity: &str,
) -> Result<(State, Store)> {
    let geometry = Geometry::candidate();
    if config.resume {
        let loaded =
            connected_checkpoint::resume(&config.out, geometry, identity, options.workers)?
                .context("connected resume requires a complete verifying checkpoint")?;
        let mut state = State::restore(&loaded.payloads, geometry, config.seed)?;
        // Paging controls are not checkpointed state; the resuming config supplies them.
        state.options = state.options.with_paging(&options);
        ensure!(
            state.anchors == loaded.anchors,
            "connected manifest/state anchor mismatch"
        );
        ensure!(
            state.options == options && loaded.store.precision == options.precision,
            "connected checkpoint options mismatch"
        );
        return Ok((state, loaded.store));
    }
    ensure!(
        connected_checkpoint::latest(&config.out)?.is_none(),
        "fresh connected run refuses checkpoints"
    );
    let state = fresh_state(geometry, config.seed, corpus, &options)?;
    let store = Store::new_precision(
        &config.out.join("connected-tiles"),
        geometry,
        config.seed,
        options.workers,
        options.precision,
    )?;
    Ok((state, store))
}

/// A fresh run's validated anchor-zero state, its readout bias the corpus log-unigram prior.
pub(super) fn fresh_state(
    geometry: Geometry,
    seed: u64,
    corpus: &Corpus,
    options: &Options,
) -> Result<State> {
    let mut state = State::new(geometry, seed, corpus.cursor(0)?, options)?;
    state.install_readout_prior(&readout_prior(corpus.byte_counts()))?;
    state.validate()?;
    Ok(state)
}

/// Log-unigram with add-1/2 (Krichevsky-Trofimov) smoothing, ln((count_b + 0.5) / (total + 128));
/// https://en.wikipedia.org/wiki/Additive_smoothing
pub(super) fn readout_prior(counts: &[u64; VOCAB]) -> Vec<f32> {
    let total = counts.iter().sum::<u64>() as f64 + 0.5 * VOCAB as f64;
    counts
        .iter()
        .map(|&count| ((count as f64 + 0.5) / total).ln() as f32)
        .collect()
}

fn checkpoint(
    config: &Config,
    state: &State,
    store: &mut Store,
    identity: &str,
) -> Result<PathBuf> {
    state.validate()?;
    store.quiescent()?;
    if let Some(path) = connected_checkpoint::latest(&config.out)? {
        if path.ends_with(format!("a{:012}", state.anchors)) {
            return Ok(path);
        }
    }
    connected_checkpoint::save(
        &config.out,
        store,
        identity,
        state.anchors,
        &state.payloads()?,
    )
}

fn evaluate(
    config: &Config,
    state: &State,
    store: &mut Store,
    corpus: &Corpus,
    log: &mut log::Log,
    retention_only: bool,
) -> Result<Value> {
    let mut evaluation = super::connected_eval::evaluate(
        state,
        store,
        corpus,
        config.eval_spans,
        &config.out,
        config.eval_cases,
        config.connected_eval.unwrap_or_default(),
        retention_only,
    )?;
    log.assess(&mut evaluation, state.anchors);
    Ok(evaluation)
}
