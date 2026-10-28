//! Core training path (`hexweave train --config`): latent Adam, greedy routes, anchor-only,
//! resident model, hash-manifested resumable checkpoints. See AGENTS.md §core.

use super::assay_source::Source as AssaySource;
use super::assay_state::MANIFEST_SHA;
use super::core_best;
use super::core_checkpoint::{self, PROTOCOL};
use super::core_drill::Drill;
use super::core_eval;
use super::core_redeal::CitationRedeal;
use super::data::Dataset;
use super::data_contracts::Split;
use super::execution::{self, TrainingSource};
use super::geometry::Geometry;
use super::resident::ResidentStore;
use super::state::State;
use super::storage_paths::digest;
use super::tuning::{self, Knobs};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
enum Data {
    /// Source-grounded QA rows (train split) with dev-split assessment.
    Science {
        root: PathBuf,
        manifest_sha256: String,
        /// Re-deal each epoch's SOURCE values and citations (`core_redeal`); absent means off.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        redeal_citations: bool,
    },
    /// Unbounded retrieval drill over the retained-text assay templates.
    Drill { root: PathBuf },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: u32,
    out: PathBuf,
    data: Data,
    seed: u64,
    rate: f64,
    anchors: u64,
    wall_seconds: u64,
    checkpoint_every: u64,
    eval_every: u64,
    eval_cases: usize,
    resume: bool,
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(self.schema == 1, "core config schema must be 1");
        let parts: Vec<Component<'_>> = self.out.components().collect();
        ensure!(
            matches!(parts.as_slice(), [Component::Normal(work), Component::Normal(runs), Component::Normal(_)]
                if *work == "work" && *runs == "core-runs"),
            "out must be work/core-runs/<name>"
        );
        ensure!(
            self.rate.is_finite() && self.rate > 0.0 && self.rate <= 1.0,
            "rate must be in (0, 1]"
        );
        ensure!(
            self.anchors > 0
                && self.checkpoint_every > 0
                && self.eval_every > 0
                && self.wall_seconds > 0,
            "anchors, checkpoint_every, eval_every and wall_seconds must be positive"
        );
        ensure!(
            (1..=32).contains(&self.eval_cases),
            "eval_cases must be in 1..=32"
        );
        Ok(())
    }

    /// Numerics identity: resuming requires the same protocol, geometry, data, seed, rate and knobs.
    fn identity(&self) -> Result<String> {
        let value = json!({ "protocol": PROTOCOL, "geometry": Geometry::candidate(),
                            "data": &self.data, "seed": self.seed, "rate": self.rate,
                            "numerics": Knobs::core(self.rate, self.seed) });
        Ok(digest(&serde_json::to_vec(&value)?))
    }
}

/// Run identity of a config document, for the identity contracts.
#[cfg(test)]
pub(super) fn identity_of(config: &[u8]) -> Result<String> {
    serde_json::from_slice::<Config>(config)?.identity()
}

/// Run (or resume) one core training job from a JSON config.
pub(super) fn run(path: &Path) -> Result<Value> {
    let config: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    config.validate()?;
    std::fs::create_dir_all(&config.out)?;
    tuning::install(Knobs::core(config.rate, config.seed));
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
                drive(&config, &source, start, evaluate)
            } else {
                drive(&config, &train, start, evaluate)
            }
        }
        Data::Drill { root } => {
            let assay = AssaySource::open(root, MANIFEST_SHA)?;
            let drill = Drill::new(&assay, config.seed)?;
            let start = State::with_cursor(geometry, config.seed, drill.cursor(0)?)?;
            drive(&config, &drill, start, &mut |state, store| {
                core_eval::drill(state, store, &assay)
            })
        }
    }
}

type Evaluate<'a> = dyn FnMut(&State, &mut ResidentStore) -> Result<Value> + 'a;

/// Interval bookkeeping between log lines.
struct Marks {
    loss: f64,
    anchors: u64,
    at: Instant,
}

impl Marks {
    fn new(state: &State) -> Self {
        Self {
            loss: state.counters.anchor_loss_sum,
            anchors: state.counters.anchors,
            at: Instant::now(),
        }
    }
}

fn record(
    log: &mut std::fs::File,
    state: &State,
    store: &mut ResidentStore,
    marks: &mut Marks,
    evaluate: &mut Evaluate<'_>,
    started: Instant,
) -> Result<Value> {
    let trained = state.counters.anchors - marks.anchors;
    let seconds = marks.at.elapsed().as_secs_f64();
    let evaluation = evaluate(state, store)?;
    let line = json!({
        "anchors": state.global_opportunity, "epoch": state.cursor.epoch,
        "train_loss": (state.counters.anchor_loss_sum - marks.loss) / trained.max(1) as f64,
        "anchors_per_second": trained as f64 / seconds.max(1e-9),
        "elapsed_seconds": started.elapsed().as_secs_f64(), "eval": evaluation,
    });
    writeln!(log, "{line}")?;
    log.flush()?;
    *marks = Marks::new(state);
    Ok(line)
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

/// Save only a Ready state whose cursor the source accepts.
fn checkpoint<S: TrainingSource>(
    config: &Config,
    source: &S,
    state: &State,
    store: &ResidentStore,
    identity: &str,
) -> Result<PathBuf> {
    state.validate_boundary()?;
    source.validate(&state.cursor)?;
    core_checkpoint::save(&config.out, state, store, identity)
}

fn drive<S: TrainingSource>(
    config: &Config,
    source: &S,
    start: State,
    evaluate: &mut Evaluate<'_>,
) -> Result<Value> {
    let started = Instant::now();
    ensure!(
        tuning::current() == Knobs::core(config.rate, config.seed),
        "core numerics profile is not installed"
    );
    let identity = config.identity()?;
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
    let mut best = core_best::best_so_far(&config.out, &identity);
    let mut last = if state.global_opportunity == 0 {
        Some(record(
            &mut log, &state, &mut store, &mut marks, evaluate, started,
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
        execution::step(&mut state, &mut store, source, false)?;
        let n = state.global_opportunity;
        if n.is_multiple_of(config.checkpoint_every) {
            checkpoint(config, source, &state, &store, &identity)?;
        }
        last = if n.is_multiple_of(config.eval_every) {
            Some(record(
                &mut log, &state, &mut store, &mut marks, evaluate, started,
            )?)
        } else {
            None
        };
        if let Some(line) = &last {
            core_best::keep(&config.out, &state, &store, &identity, line, &mut best);
        }
    };
    let n = state.global_opportunity;
    let checkpoint = match core_checkpoint::latest(&config.out)? {
        Some(path) if path.ends_with(format!("a{n:012}")) => path,
        _ => checkpoint(config, source, &state, &store, &identity)?,
    };
    let last = match last {
        Some(line) => line,
        None => record(&mut log, &state, &mut store, &mut marks, evaluate, started)?,
    };
    core_best::keep(&config.out, &state, &store, &identity, &last, &mut best);
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
