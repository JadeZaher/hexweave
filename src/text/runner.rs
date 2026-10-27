use super::credit::{anchor, AnchorRequest};
use super::data::{example, Corpus};
use super::execution::{evaluate, execute, Counts, ForwardRequest};
use super::history::History;
use super::report::{report, Timings};
use super::shared::Shared;
use super::storage::{write_new, Store};
use super::{CreditDiagnostics, Mode, Options, VOCAB, WIDTH};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

pub(super) fn run(options: &Options, data: &Corpus, directory: &Path) -> Result<Value> {
    let started = Instant::now();
    let mut shared = Shared::new(options.seed)?;
    let mut store = Store::create(directory, options.seed)?;
    let mut timings = Timings {
        initialization_seconds: started.elapsed().as_secs_f64(),
        ..Timings::default()
    };
    let evaluation_started = Instant::now();
    let (initial_loss, evaluation_examples) = evaluate(&shared, &mut store, &data.valid)?;
    timings.initial_evaluation_seconds = evaluation_started.elapsed().as_secs_f64();
    let mut counts = Counts {
        credit: (options.credit_diagnostics == CreditDiagnostics::On).then(Default::default),
        ..Counts::default()
    };
    let mut history = History::default();
    let mut baseline = (VOCAB as f32).ln();
    let mut completed = 0;
    let mut stopped = false;
    let training_started = Instant::now();
    for step in 0..options.steps {
        if store.directory.join("STOP").exists() {
            stopped = true;
            break;
        }
        train_step(
            options,
            data,
            &mut shared,
            &mut store,
            &mut history,
            &mut counts,
            &mut baseline,
            step,
        )?;
        completed = step + 1;
        if completed.is_multiple_of(32) || completed == options.steps {
            progress(
                &store.directory,
                completed,
                &counts,
                training_started.elapsed().as_secs_f64(),
            )?;
        }
    }
    timings.training_seconds = training_started.elapsed().as_secs_f64();
    let evaluation_started = Instant::now();
    let final_loss = if stopped {
        None
    } else {
        Some(evaluate(&shared, &mut store, &data.valid)?.0)
    };
    timings.final_evaluation_seconds = evaluation_started.elapsed().as_secs_f64();
    let snapshot_started = Instant::now();
    write_snapshot(options, data, &store, &shared, &counts, completed, stopped)?;
    timings.snapshot_seconds = snapshot_started.elapsed().as_secs_f64();
    timings.total_seconds = started.elapsed().as_secs_f64();
    let result = report(
        options,
        data,
        &store,
        &shared,
        &history,
        &counts,
        &timings,
        completed,
        stopped,
        initial_loss,
        final_loss,
        evaluation_examples,
    )?;
    write_new(
        &store.directory.join("result.json"),
        &serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}

fn write_snapshot(
    options: &Options,
    data: &Corpus,
    store: &Store,
    shared: &Shared,
    counts: &Counts,
    completed: usize,
    stopped: bool,
) -> Result<()> {
    write_new(
        &store.directory.join("shared-final.json"),
        &serde_json::to_vec(&shared.snapshot())?,
    )?;
    let snapshot = json!({
        "schema": 4, "protocol": "text-pulse-v4", "status": if stopped { "stopped" } else { "complete" },
        "tile_versions": store.versions, "completed_step_opportunities": completed,
        "anchor_targets": counts.anchor_targets, "options": options, "data": data.identity(),
        "shared_identity_fnv1a64": format!("{:016x}", shared.identity()?),
        "resumable": false, "reason": "inspection snapshot; no optimizer/history/cursor restore protocol"
    });
    write_new(
        &store.directory.join("snapshot.json"),
        &serde_json::to_vec(&snapshot)?,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train_step(
    options: &Options,
    data: &Corpus,
    shared: &mut Shared,
    store: &mut Store,
    history: &mut History,
    counts: &mut Counts,
    baseline: &mut f32,
    step: usize,
) -> Result<()> {
    let is_anchor = step.is_multiple_of(options.anchor_interval);
    if !is_anchor && options.mode == Mode::AnchorOnly {
        return Ok(());
    }
    history.prune(&store.versions, shared.version, step);
    if !is_anchor {
        if let Some(record) = history.latest() {
            counts.predictor_error_sum +=
                shared.train_predictor(&record.features, &record.correction)?;
            counts.predictor_updates += 1;
            counts.predictor_replays += 1;
        }
    }
    let signal = if options.mode == Mode::Pulse {
        history.summary()
    } else {
        vec![0.0; WIDTH]
    };
    let example_index = if is_anchor {
        step / options.anchor_interval
    } else {
        step + 17
    };
    let context = super::data::context(&data.train, example_index);
    let frames = execute(
        shared,
        store,
        history,
        counts,
        ForwardRequest {
            context,
            history: &signal,
            random_seed: Some(options.seed.wrapping_add(step as u64 * 7919)),
            local: !is_anchor,
            step,
        },
    )?;
    if is_anchor {
        let loss = anchor(
            shared,
            store,
            history,
            counts,
            AnchorRequest {
                context,
                target: example(&data.train, example_index).1,
                signal: &signal,
                frames: &frames,
                baseline: *baseline,
                step,
                predictor: options.mode == Mode::Pulse,
                route_credit: options.route_credit,
            },
        )?;
        *baseline = 0.9 * *baseline + 0.1 * loss;
        ensure!(baseline.is_finite(), "nonfinite baseline");
    }
    Ok(())
}

fn progress(directory: &Path, completed: usize, counts: &Counts, seconds: f64) -> Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(directory.join("progress.jsonl"))?;
    let event = json!({"completed_step_opportunities": completed, "anchor_targets": counts.anchor_targets,
        "tile_visits": counts.tile_visits, "training_seconds": seconds});
    writeln!(file, "{}", serde_json::to_string(&event)?)?;
    file.flush()?;
    Ok(())
}
