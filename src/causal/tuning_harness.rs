//! Ignored in-memory tuning harness over the retained-text assay; see AGENTS.md §tuning.

use super::assay_observation;
use super::assay_source::{cursor, Source};
use super::assay_state::{Progress, MANIFEST_SHA, PASSES};
use super::assay_update::Rule;
use super::execution;
use super::generation::Terminal;
use super::geometry::Geometry;
use super::state::State;
use super::store_api::TileStore;
use super::tuning::{self, Knobs, Update};
use super::tuning_measure::{attention_mass, evaluate};
use super::tuning_sources::Shuffled;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

/// Unset is None; a set but unparseable variable is an error, never a silent pinned fallback.
fn env<T: std::str::FromStr>(name: &str) -> Result<Option<T>> {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .map(Some)
            .map_err(|_| anyhow::anyhow!("unparseable {name}={value}")),
        Err(_) => Ok(None),
    }
}

fn knobs(seed: u64) -> Result<Knobs> {
    let update = match env::<String>("HEXWEAVE_TUNE_UPDATE")?.as_deref() {
        None | Some("normalized") => Update::Normalized,
        Some("raw") => Update::Raw,
        Some("latent") => Update::Latent,
        Some(other) => anyhow::bail!("unknown HEXWEAVE_TUNE_UPDATE={other}"),
    };
    Ok(Knobs {
        tile_rate: env("HEXWEAVE_TUNE_TILE_RATE")?,
        router_rate: env("HEXWEAVE_TUNE_ROUTER_RATE")?,
        embed_rate: env("HEXWEAVE_TUNE_EMBED_RATE")?,
        update,
        temperature: env("HEXWEAVE_TUNE_TEMPERATURE")?.unwrap_or(1.0),
        attention_scale: env("HEXWEAVE_TUNE_ATTENTION_SCALE")?.unwrap_or(1.0),
        attention_mix: env("HEXWEAVE_TUNE_ATTENTION_MIX")?.unwrap_or(0.25),
        greedy_routes: env("HEXWEAVE_TUNE_GREEDY_ROUTES")?.unwrap_or(false),
        no_halt: env("HEXWEAVE_TUNE_NO_HALT")?.unwrap_or(false),
        route_salt: 0,
        seed,
        float_forward: false,
        history: true,
    })
}

/// Early passes, the end of each foundation phase (16) and each segment (20), plus every Nth global pass.
fn due(segment: u64, local: u64, every: u64) -> bool {
    local == 16
        || local == 20
        || (segment == 0 && matches!(local, 0 | 1 | 2 | 4 | 8))
        || (segment * 20 + local).is_multiple_of(every)
}

/// Publish the production assay observation the ledger requires at this boundary.
fn record(
    state: &mut State,
    store: &mut fixture::Store,
    source: &Source,
    directory: &Path,
) -> Result<()> {
    let recorded = state
        .assay
        .as_ref()
        .context("assay state")?
        .observations
        .len();
    if PASSES.get(recorded) != Some(&(state.global_opportunity / 80))
        || !state.global_opportunity.is_multiple_of(80)
    {
        return Ok(());
    }
    let mut never = || -> Result<Option<Terminal>> { Ok(None) };
    let observation = assay_observation::observe(state, store, source, &mut never)?;
    let reference = assay_observation::publish(directory, &observation)?;
    state
        .assay
        .as_mut()
        .context("assay state")?
        .observations
        .push(reference);
    Ok(())
}

fn fresh(seed: u64, rule: Rule) -> Result<State> {
    let mut state = State::with_cursor(Geometry::candidate(), seed, cursor(0)?)?;
    state.assay = Some(Progress::new(rule));
    state.validate_boundary()?;
    Ok(state)
}

/// Continue past the fixed course: learned E and tiles enter a fresh assay ledger at version 0.
fn rollover(state: State, store: &mut fixture::Store, rule: Rule) -> Result<State> {
    let mut next = fresh(state.seed, rule)?;
    next.shared = state.shared;
    next.shared.version = 0;
    next.baseline = state.baseline;
    for tile in &mut store.tiles {
        tile.version = 0;
    }
    next.tile_versions = store.versions()?;
    next.validate_boundary()?;
    Ok(next)
}

/// Interval accounting since the previous evaluation line.
#[derive(Default)]
struct Marks {
    loss: f64,
    anchors: u64,
    shared: u64,
    tiles: u64,
}

fn line(state: &State, marks: &mut Marks, segment: u64, started: Instant) -> Value {
    let counters = &state.counters;
    let tiles: u64 = counters.tile_transitions.iter().sum();
    let trained = counters.anchors - marks.anchors;
    let value = json!({
        "knobs": format!("{:?}", tuning::current()), "seed": state.seed, "segment": segment,
        "pass": segment * 20 + state.global_opportunity / 80,
        "segment_anchors": state.global_opportunity,
        "train_loss": (counters.anchor_loss_sum - marks.loss) / trained.max(1) as f64,
        "shared_transitions": counters.shared_transitions - marks.shared,
        "tile_transitions": tiles - marks.tiles, "interval_anchors": trained,
        "elapsed_s": started.elapsed().as_secs_f64(),
    });
    *marks = Marks {
        loss: counters.anchor_loss_sum,
        anchors: counters.anchors,
        shared: counters.shared_transitions,
        tiles,
    };
    value
}

#[test]
#[ignore = "bounded tuning run; driven by HEXWEAVE_TUNE_* variables"]
fn tuning_run() -> Result<()> {
    let seed = env("HEXWEAVE_TUNE_SEED")?.unwrap_or(7);
    let knobs = knobs(seed)?;
    let rule = match env::<String>("HEXWEAVE_TUNE_RULE")?.as_deref() {
        _ if knobs.embed_rate.is_some() => Rule::Magnitude,
        Some("magnitude") => Rule::Magnitude,
        _ => Rule::Dense,
    };
    let segments: u64 = env::<u64>("HEXWEAVE_TUNE_SEGMENTS")?.unwrap_or(1).max(1);
    let every: u64 = env::<u64>("HEXWEAVE_TUNE_EVAL_EVERY")?.unwrap_or(4).max(1);
    let decay: f64 = env("HEXWEAVE_TUNE_DECAY")?.unwrap_or(1.0);
    let shuffle: bool = env("HEXWEAVE_TUNE_SHUFFLE")?.unwrap_or(false);
    let output = PathBuf::from(env::<String>("HEXWEAVE_TUNE_OUT")?.context("HEXWEAVE_TUNE_OUT")?);
    let _pinned_on_exit = tuning::scoped(knobs);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/retained-text-assay");
    let source = Source::open(&root, MANIFEST_SHA)?;
    let mut state = fresh(seed, rule)?;
    let mut store = fixture::Store::with_geometry(Geometry::candidate(), seed)?;
    let mut file = std::fs::File::create(&output)?;
    let (started, mut marks) = (Instant::now(), Marks::default());
    for segment in 0..segments {
        if segment > 0 {
            state = rollover(state, &mut store, rule)?;
            marks = Marks::default();
            let factor = decay.powi(i32::try_from(segment)?);
            tuning::anneal(Knobs {
                tile_rate: knobs.tile_rate.map(|rate| rate * factor),
                router_rate: knobs.router_rate.map(|rate| rate * factor),
                embed_rate: knobs.embed_rate.map(|rate| rate * factor),
                route_salt: segment.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                ..knobs
            });
        }
        let directory = output
            .with_extension("obs")
            .join(format!("segment-{segment}"));
        std::fs::create_dir_all(&directory)?;
        let shuffled = Shuffled {
            inner: &source,
            salt: tuning::hash(seed, &[segment]),
        };
        loop {
            let n = state.global_opportunity;
            if n.is_multiple_of(80) && (n > 0 || segment == 0) && due(segment, n / 80, every) {
                let mut value = line(&state, &mut marks, segment, started);
                value["shuffle"] = json!(shuffle);
                value["attention"] = attention_mass(&state, &mut store, &source)?;
                value["cases"] = evaluate(&state, &mut store, &source)?;
                writeln!(file, "{value}")?;
                file.flush()?;
            }
            record(&mut state, &mut store, &source, &directory)?;
            if n >= 1600 {
                break;
            }
            if shuffle {
                execution::step(&mut state, &mut store, &shuffled, false)?;
            } else {
                execution::step(&mut state, &mut store, &source, false)?;
            }
        }
    }
    Ok(())
}
