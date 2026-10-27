//! Ignored bounded persistence-overhead harness over real steps and DiskStore; see AGENTS.md §ledger.

use super::assay_source::{cursor, Source};
use super::assay_state::{Progress, CONTRACT_SHA, PASSES, PROTOCOL};
use super::assay_update::Rule;
use super::checkpoint::publish;
use super::checkpoint_retention::{retain, RECEIPT_GENERATIONS};
use super::checkpoint_state::Identity;
use super::execution::{self, TrainingSource};
use super::geometry::Geometry;
use super::profile::{self, Bucket, Totals, BUCKETS};
use super::state::State;
use super::storage::DiskStore;
use anyhow::Result;
use std::time::{Duration, Instant};

#[path = "persistence_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

const NAMES: [&str; BUCKETS] = [
    "store_read",
    "store_write",
    "dir_scan",
    "publish_io",
    "fsync",
    "source_verify",
    "source_guard",
];
const DISK: u64 = 64 * 1024 * 1024;
const EVERY: u64 = 64;

#[derive(Default)]
struct Run {
    steps: Duration,
    totals: Totals,
    checkpoint: Duration,
    checkpoints: u32,
    observation: Duration,
    opportunities: u64,
    files: usize,
}

fn opportunities(default: u64) -> u64 {
    std::env::var("HEXWEAVE_PROFILE_OPPORTUNITIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .max(EVERY)
}

fn save<S: TrainingSource>(
    store: &mut DiskStore,
    state: &State,
    identity: &Identity,
    source: &S,
    run: &mut Run,
) -> Result<()> {
    let mut owner = fixture::owner(state, "profile");
    owner.budget.maximum_opportunities = 1601;
    owner.budget.disk_bytes_per_attempt = DISK;
    let began = Instant::now();
    publish(store, state, identity, &owner, source)?;
    retain(store, RECEIPT_GENERATIONS)?;
    run.checkpoint += began.elapsed();
    run.checkpoints += 1;
    Ok(())
}

fn drive<S: TrainingSource>(
    state: &mut State,
    store: &mut DiskStore,
    source: &S,
    identity: &Identity,
    count: u64,
    observe: &mut dyn FnMut(&mut State, &mut DiskStore) -> Result<bool>,
) -> Result<Run> {
    let mut run = Run::default();
    save(store, state, identity, source, &mut run)?;
    for n in 1..=count {
        let began = Instant::now();
        if observe(state, store)? {
            run.observation += began.elapsed();
            save(store, state, identity, source, &mut run)?;
        }
        let before = profile::snapshot();
        let began = Instant::now();
        execution::step(state, store, source, false)?;
        run.steps += began.elapsed();
        let delta = profile::snapshot().since(&before);
        for i in 0..BUCKETS {
            run.totals.nanos[i] += delta.nanos[i];
            run.totals.calls[i] += delta.calls[i];
        }
        run.opportunities = n;
        if n.is_multiple_of(EVERY) {
            save(store, state, identity, source, &mut run)?;
        }
    }
    run.files = std::fs::read_dir(&store.directory)?.count();
    Ok(run)
}

fn print(label: &str, run: &Run) {
    let n = run.opportunities as f64;
    let ms = |nanos: u64| nanos as f64 / 1e6;
    let steps = run.steps.as_nanos() as u64;
    println!(
        "== {label}: {} opportunities, {} checkpoint+retain, {} files at end",
        run.opportunities, run.checkpoints, run.files
    );
    println!(
        "{:<16}{:>12}{:>14}{:>12}",
        "bucket", "calls/opp", "total_ms", "ms/opp"
    );
    println!(
        "{:<16}{:>12.2}{:>14.1}{:>12.3}",
        "step_total",
        1.0,
        ms(steps),
        ms(steps) / n
    );
    for (i, name) in NAMES.iter().enumerate() {
        let calls = run.totals.calls[i] as f64 / n;
        let total = ms(run.totals.nanos[i]);
        println!("{name:<16}{calls:>12.2}{total:>14.1}{:>12.3}", total / n);
    }
    let outside = [
        Bucket::StoreRead,
        Bucket::StoreWrite,
        Bucket::SourceVerify,
        Bucket::SourceGuard,
    ]
    .iter()
    .map(|&bucket| run.totals.nanos[bucket as usize])
    .sum::<u64>();
    let compute = ms(steps.saturating_sub(outside));
    println!(
        "{:<16}{:>12}{:>14.1}{:>12.3}",
        "compute_resid",
        "-",
        compute,
        compute / n
    );
    let observation = run.observation.as_secs_f64() * 1e3;
    println!("observation (outside steps): {observation:.1} ms");
    let checkpoint = run.checkpoint.as_secs_f64() * 1e3;
    println!(
        "checkpoint+retain: {checkpoint:.1} ms total, {:.1} ms each",
        checkpoint / f64::from(run.checkpoints)
    );
}

#[test]
#[ignore = "bounded persistence profile; run explicitly with --nocapture"]
fn persistence_overhead_profile() -> Result<()> {
    let f = fixture::Fixture::new()?;
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/retained-text-assay");
    let source = Source::open(&root, super::assay_state::MANIFEST_SHA)?;
    let mut state = State::with_cursor(Geometry::candidate(), 7, cursor(0)?)?;
    state.assay = Some(Progress::new(Rule::Dense));
    state.validate_boundary()?;
    let mut identity = fixture::identity(&state)?;
    identity.protocol = PROTOCOL.into();
    identity.protocol_sha256 = CONTRACT_SHA.into();
    identity.config_sha256 = super::config::Config::assay(7, Rule::Dense).sha256()?;
    let attempt = f.attempt("profile")?;
    let mut store = DiskStore::create(&f.workspace, &attempt, state.geometry, 7, DISK)?;
    let mut observe = |state: &mut State, store: &mut DiskStore| -> Result<bool> {
        let progress = state.assay.as_ref().unwrap();
        let due = PASSES.get(progress.observations.len());
        if due.is_none_or(|pass| pass * 80 != state.global_opportunity) {
            return Ok(false);
        }
        let observation =
            super::assay_observation::observe(state, store, &source, &mut || Ok(None))?;
        let reference = super::assay_observation::publish(&attempt, &observation)?;
        state.assay.as_mut().unwrap().observations.push(reference);
        Ok(true)
    };
    let count = opportunities(128);
    let assay = drive(
        &mut state,
        &mut store,
        &source,
        &identity,
        count,
        &mut observe,
    )?;
    print("assay candidate seed7 dense", &assay);
    drop(store);
    let data = super::data_tests::Fixture::new();
    let dataset = data.open()?;
    let mut state = State::new(Geometry::candidate(), 7, &dataset)?;
    let identity = fixture::identity(&state)?;
    let g = fixture::Fixture::new()?;
    let attempt = g.attempt("profile")?;
    let mut store = DiskStore::create(&g.workspace, &attempt, state.geometry, 7, DISK)?;
    let v3 = drive(
        &mut state,
        &mut store,
        &dataset,
        &identity,
        EVERY,
        &mut |_, _| Ok(false),
    )?;
    print("v3 dataset candidate seed7", &v3);
    drop(store);
    std::fs::remove_dir_all(&f.root)?;
    std::fs::remove_dir_all(&g.root)?;
    std::fs::remove_dir_all(&data.root)?;
    Ok(())
}
