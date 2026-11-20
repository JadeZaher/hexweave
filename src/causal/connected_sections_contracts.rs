//! Shared-section flow: exact resume, identity refusal, monitors, entry credit and causality.

use super::super::connected_anchor::{self, Credit, Trace};
use super::super::connected_checkpoint as checkpoint;
use super::super::connected_eval;
use super::super::connected_route::Draw;
use super::super::connected_sections_read::{entry, observe};
use super::super::connected_sections_write::{finish, update};
use super::super::connected_step;
use super::super::connected_storage_contracts::fresh;
use super::super::connected_store::Store;
use super::super::connected_wave::step;
use super::super::core_corpus::{Corpus, END};
use super::super::shared::Prefix;
use super::super::storage_paths::digest;
use super::super::tuning::{self, Knobs};
use super::rules::config;
use super::*;
use serde_json::{json, Value};
use std::path::Path;

/// Anchor-exact FP32 options with a Lambert restart block and `shared`, cap 2.
fn options(shared: Value, batch: u32) -> Result<Options> {
    Ok(serde_json::from_value(json!({
        "precision":"fp32","lanes":2,"workers":2,"visits":4,"credit_mode":"anchor_exact",
        "anchor_batch":batch,"shared":shared,
        "restart":{"floor":0.1,"threshold":0.5,"signal":"tile","law":"lambert"},
    }))?)
}

fn arm(root: &Path, corpus: &Corpus, options: Options) -> Result<(State, Store)> {
    let geometry = Geometry::fixture();
    let state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let tiles = root.join("connected-tiles");
    let mut store = Store::new_precision(&tiles, geometry, 7, options.workers, options.precision)?;
    store.set_paging(options.resident(), 0)?;
    Ok((state, store))
}

fn train(state: &mut State, store: &mut Store, corpus: &Corpus, anchors: u64) -> Result<()> {
    let target = state.anchors + anchors;
    while state.anchors < target {
        step(state, store, corpus, state.options, 0.003, false)?;
    }
    state.validate()
}

/// A fixture corpus over `train` whose held-out story ends with `END` (evaluation samples need it).
fn corpus(root: &Path, train: &[u8]) -> Result<Corpus> {
    let heldout = [&b"A small story."[..], &[END]].concat();
    let entry = |bytes: &[u8]| json!({"bytes":bytes.len(),"sha256":digest(bytes)});
    let manifest = serde_json::to_vec(&json!({"schema":1,"kind":"byte-corpus",
        "files":{"train.txt":entry(train),"heldout.txt":entry(&heldout)}}))?;
    std::fs::create_dir_all(root)?;
    std::fs::write(root.join("train.txt"), train)?;
    std::fs::write(root.join("heldout.txt"), &heldout)?;
    std::fs::write(root.join("manifest.json"), &manifest)?;
    let mut corpus = Corpus::open(root, &digest(&manifest), 32)?;
    corpus.window = Geometry::fixture().context;
    Ok(corpus)
}

fn text() -> Vec<u8> {
    let cycle = b"the cat sat on the mat. ".iter().cycle();
    cycle.take(512).copied().collect()
}

fn scoped() -> tuning::Scoped {
    tuning::scoped(Knobs {
        connected: true,
        connected_fp32: true,
        ..Knobs::core(0.003, 7)
    })
}

#[test]
fn shared_waves_resume_exactly_refuse_other_settings_and_report_monitors() -> Result<()> {
    let _guard = scoped();
    let out = fresh("shared-sections")?;
    let corpus = corpus(&out.join("corpus"), &text())?;
    let shared = json!({"sections":3,"wave":4,"teach":0.25});
    let options = options(shared.clone(), 2)?;
    let (straight, mut straight_store) = {
        let (mut state, mut store) = arm(&out.join("straight"), &corpus, options)?;
        train(&mut state, &mut store, &corpus, 12)?;
        (state, store)
    };
    let sections = straight.sections.as_ref().expect("sections");
    assert_eq!(sections.queue.waves, 3);
    // Exactly the written sections left their seeded masters.
    let seeded = super::fresh(Geometry::fixture(), 7, &options)?.expect("sections");
    let pairs = sections.sections.iter().zip(&seeded.sections);
    for ((section, start), slot) in pairs.zip(&sections.queue.slots) {
        assert_eq!(section.latents != start.latents, slot.writes > 0);
    }
    let report = report(&straight)?.expect("shared report");
    // The read must differ from the entry state it is added to (a section is a residual tile).
    let delta = report["read_delta"].as_f64().expect("read delta");
    assert!(delta.is_finite() && delta > 0.0, "{report}");
    let weights = report["mean_tile_weight"].as_array().expect("tile weights");
    assert!(weights
        .iter()
        .all(|w| w.is_null() || w.as_f64().is_some_and(|w| (0.0..=1.0).contains(&w))));
    // A checkpoint at a shared-wave end resumes bit-exactly; another `shared` identity is refused.
    let run = out.join("run");
    let identity = |shared: Value| config(json!({ "shared": shared }))?.identity();
    let (mine, other) = (identity(shared)?, identity(json!({"sections":2,"wave":4}))?);
    let (mut first, mut first_store) = arm(&run, &corpus, options)?;
    train(&mut first, &mut first_store, &corpus, 8)?;
    checkpoint::save(&run, &mut first_store, &mine, 8, &first.payloads()?)?;
    drop((first, first_store));
    let refused = checkpoint::resume(&run, Geometry::fixture(), &other, 2);
    assert!(refused.is_err_and(|error| error.to_string().contains("foreign")));
    let loaded = checkpoint::resume(&run, Geometry::fixture(), &mine, 2)?.expect("checkpoint");
    let mut resumed = State::restore(&loaded.payloads, Geometry::fixture(), 7)?;
    let mut store = loaded.store;
    store.set_paging(2, 0)?;
    train(&mut resumed, &mut store, &corpus, 4)?;
    assert_eq!(resumed.payloads()?, straight.payloads()?);
    store.flush()?;
    straight_store.flush()?;
    let meta = |store: &Store| {
        store
            .slots
            .iter()
            .map(|s| s.meta.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(meta(&store), meta(&straight_store));
    // Frozen evaluation reads the sections and reports both monitors without mutating anything.
    let assessment = super::super::connected_eval_config::Options {
        budget_seconds: 30,
        retention_spans: 0,
        sample_bytes: 1,
    };
    let evaluated =
        connected_eval::evaluate(&resumed, &mut store, &corpus, 1, &run, 1, assessment, false)?;
    let monitors = &evaluated["shared"];
    assert!(
        monitors["heldout_without_shared"]
            .as_f64()
            .is_some_and(f64::is_finite),
        "{monitors}"
    );
    let similarity = monitors["tile_similarity"]
        .as_f64()
        .expect("tile similarity");
    // Distinct seeded tiles change the same states differently; identical ones would give 1.
    assert!(similarity > -1.0 && similarity < 1.0 - 1e-6, "{similarity}");
    Ok(())
}

#[test]
fn shared_predictions_ignore_labels_at_and_after_their_position() -> Result<()> {
    let _guard = scoped();
    let out = fresh("shared-causal")?;
    let options = options(json!({"sections":2,"wave":4,"teach":0.25}), 1)?;
    // Per-anchor losses of 8 anchors (two shared waves) on `train`.
    let losses = |name: &str, train: &[u8]| -> Result<Vec<f32>> {
        let corpus = corpus(&out.join(name), train)?;
        let (mut state, mut store) = arm(&out.join(format!("{name}-run")), &corpus, options)?;
        let mut losses = Vec::new();
        for _ in 0..8 {
            let loss = connected_step::step(&mut state, &mut store, &corpus, options, 0.003, false);
            losses.push(loss?);
            if state.anchors.is_multiple_of(4) {
                finish(&mut state, &options)?;
            }
        }
        Ok(losses)
    };
    let original = text();
    let reference = losses("original", &original)?;
    // Changes from the last anchor of a wave, its end, and inside the next wave.
    for changed in [3usize, 4, 5, 7] {
        let mut altered = original.clone();
        altered[changed..].iter_mut().for_each(|byte| *byte = b'z');
        let probe = losses(&format!("changed-{changed}"), &altered)?;
        assert_eq!(
            probe[..changed],
            reference[..changed],
            "labels from {changed}"
        );
        assert_ne!(probe, reference, "the change at {changed} must matter");
    }
    Ok(())
}

/// The fixed one-visit anchor "the cat" -> ' ' under zero history: prefix, trace and credit.
fn fixed(state: &State, store: &mut Store) -> Result<(Prefix, Trace, Credit)> {
    let prefix = state.shared.prefix(b"the cat")?;
    let zeros = vec![0.0; state.geometry.width];
    let start = entry(state.sections.as_ref(), state.geometry, &prefix)?;
    let (geometry, visits) = (state.geometry, state.options.visits);
    let history = |_: usize, _: &[f32]| Ok(zeros.clone());
    let trace = connected_anchor::forward_from(
        store,
        geometry,
        &prefix,
        start,
        visits,
        Draw::GREEDY,
        history,
    )?;
    let credit = connected_anchor::gather(&state.shared, store, &prefix, &trace, b' ')?;
    Ok((prefix, trace, credit))
}

#[test]
fn entry_credit_write_lowers_the_loss_on_its_own_input() -> Result<()> {
    let _guard = scoped();
    let out = fresh("shared-entry")?;
    let corpus = corpus(&out.join("corpus"), &text())?;
    // One visit: no route choice, so the write alone moves the loss.
    let mut options = options(json!({"sections":1,"wave":1,"credit":"entry"}), 1)?;
    (options.visits, options.local_rate) = (1, 1e-3);
    let (mut state, mut store) = arm(&out.join("run"), &corpus, options)?;
    tuning::set_connected(&options);
    let (prefix, trace, mut before) = fixed(&state, &mut store)?;
    observe(&mut state, &mut store, &prefix, &trace, &mut before)?;
    let sections = state.sections.as_mut().expect("sections");
    let item = sections.pending.pop().expect("queued write credit");
    update(
        &mut sections.sections[0],
        &[item],
        options.local_rate,
        &options,
    )?;
    let (_, _, after) = fixed(&state, &mut store)?;
    assert!(
        after.loss < before.loss,
        "{} -> {}",
        before.loss,
        after.loss
    );
    Ok(())
}
