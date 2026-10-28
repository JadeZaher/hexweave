//! Core-path contracts: harness parity, drill stream and run identity; see AGENTS.md §core.

use super::assay_observation;
use super::assay_source::{cursor, Source};
use super::assay_state::{Progress, MANIFEST_SHA};
use super::assay_update::Rule;
use super::core_codec::encode_latents;
use super::core_config::identity_of;
use super::core_corpus::{window_before, Corpus, START};
use super::core_drill::{Drill, CYCLE};
use super::execution::{self, TrainingSource};
use super::generation::Terminal;
use super::geometry::Geometry;
use super::resident::ResidentStore;
use super::state::State;
use super::state_codec::{encode_predictor, encode_shared, encode_tile};
use super::storage_paths::digest;
use super::tuning::{self, Knobs};
use super::tuning_sources::Shuffled;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::path::Path;

pub(super) fn assay() -> Result<Source> {
    Source::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/retained-text-assay"),
        MANIFEST_SHA,
    )
}

/// Everything training numerics touch: tiles, E, predictor, latents, history, baseline, loss.
pub(super) fn fingerprint(state: &State, store: &ResidentStore) -> Result<Vec<Vec<u8>>> {
    let mut parts: Vec<Vec<u8>> = store.tiles.iter().map(encode_tile).collect::<Result<_>>()?;
    let (latents, calls) = tuning::export();
    parts.extend([
        encode_shared(&state.shared)?,
        encode_predictor(&state.predictor)?,
        encode_latents(&latents)?,
        calls.to_le_bytes().to_vec(),
        serde_json::to_vec(&state.history)?,
        state.baseline.to_bits().to_le_bytes().to_vec(),
        state
            .counters
            .anchor_loss_sum
            .to_bits()
            .to_le_bytes()
            .to_vec(),
    ]);
    Ok(parts)
}

/// Per-process, per-test temporary directory, cleared of any leftover from an earlier run.
pub(super) fn fresh_temp(test: &str) -> Result<std::path::PathBuf> {
    let path = std::env::temp_dir().join(format!("hexweave-core-{test}-{}", std::process::id()));
    if path.exists() {
        std::fs::remove_dir_all(&path)?;
    }
    Ok(path)
}

pub(super) fn train<S: TrainingSource>(
    state: &mut State,
    store: &mut ResidentStore,
    source: &S,
    anchors: u64,
) -> Result<()> {
    for _ in 0..anchors {
        execution::step(state, store, source, false)?;
    }
    Ok(())
}

/// The harness configuration that learned retrieval: `Shuffled` under the assay ledger.
fn harness_fingerprint(source: &Source, geometry: Geometry, anchors: u64) -> Result<Vec<Vec<u8>>> {
    let _core = tuning::scoped(Knobs::core(0.003, 7));
    let shuffled = Shuffled {
        inner: source,
        salt: tuning::hash(7, &[0]),
    };
    let mut state = State::with_cursor(geometry, 7, cursor(0)?)?;
    state.assay = Some(Progress::new(Rule::Magnitude));
    let mut store = ResidentStore::new(geometry, 7)?;
    let out = fresh_temp("parity")?;
    std::fs::create_dir_all(&out)?;
    let mut never = || -> Result<Option<Terminal>> { Ok(None) };
    let observation = assay_observation::observe(&state, &mut store, source, &mut never)?;
    let reference = assay_observation::publish(&out, &observation)?;
    state
        .assay
        .as_mut()
        .context("assay state")?
        .observations
        .push(reference);
    train(&mut state, &mut store, &shuffled, anchors)?;
    let parts = fingerprint(&state, &store)?;
    std::fs::remove_dir_all(&out)?;
    Ok(parts)
}

#[test]
fn core_drill_reproduces_validated_harness_numerics() -> Result<()> {
    let source = assay()?;
    let geometry = Geometry::candidate();
    let harness = harness_fingerprint(&source, geometry, 12)?;
    let _core = tuning::scoped(Knobs::core(0.003, 7));
    let drill = Drill::new(&source, 7)?;
    let mut state = State::with_cursor(geometry, 7, drill.cursor(0)?)?;
    state.anchor_only = true;
    let mut store = ResidentStore::new(geometry, 7)?;
    train(&mut state, &mut store, &drill, 12)?;
    let core = fingerprint(&state, &store)?;
    assert!(
        harness == core,
        "production drill diverges from the harness at fingerprint part {:?}",
        harness.iter().zip(&core).position(|(a, b)| a != b)
    );
    assert_eq!(
        state.numerical_clock()?,
        4 * 12,
        "anchor-only clock is 4x physical work"
    );
    Ok(())
}

/// The persisted run identity: a golden digest pins the formula, and only numerics fields move it.
#[test]
fn run_identity_is_pinned_and_ignores_schedule_fields() -> Result<()> {
    let base = json!({"schema": 1, "out": "work/core-runs/x", "seed": 7, "rate": 0.003,
        "data": {"kind": "drill", "root": "datasets/retained-text-assay"}, "anchors": 1,
        "wall_seconds": 1, "checkpoint_every": 1, "eval_every": 1, "eval_cases": 1, "resume": false});
    let identity = |field: &str, value: Value| -> Result<String> {
        let mut config = base.clone();
        config[field] = value;
        identity_of(&serde_json::to_vec(&config)?)
    };
    let reference = identity("schema", json!(1))?;
    assert_eq!(
        reference, "a32ed6d967e77736ce19b3a0a738c4b33f75b4e22318f32d6a55b204972d98ff",
        "identity formula changed: existing checkpoints will not resume"
    );
    for (field, value, moves) in [
        ("seed", json!(8), true),
        ("rate", json!(0.004), true),
        (
            "data",
            json!({"kind": "drill", "root": "datasets/other"}),
            true,
        ),
        ("anchors", json!(2), false),
        ("wall_seconds", json!(9), false),
        ("checkpoint_every", json!(5), false),
        ("eval_every", json!(5), false),
        ("eval_cases", json!(3), false),
        ("resume", json!(true), false),
        ("out", json!("work/core-runs/y"), false),
    ] {
        assert_eq!(identity(field, value)? != reference, moves, "{field}");
    }
    let science = |redeal: Option<bool>| {
        let mut data = json!({"kind": "science", "root": "r", "manifest_sha256": "m"});
        if let Some(on) = redeal {
            data["redeal_citations"] = json!(on);
        }
        identity("data", data)
    };
    assert_eq!(
        science(None)?,
        science(Some(false))?,
        "absent re-deal keeps old identities"
    );
    assert_ne!(
        science(None)?,
        science(Some(true))?,
        "re-deal is part of the run identity"
    );
    Ok(())
}

#[test]
fn drill_stream_is_the_harness_stream_and_answers_need_the_key() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    for n in [
        0,
        4,
        79,
        1_284,
        CYCLE - 1,
        CYCLE,
        CYCLE + 7,
        30 * CYCLE + 1_283,
    ] {
        let shuffled = Shuffled {
            inner: &source,
            salt: tuning::hash(7, &[n / CYCLE]),
        };
        let (at, local) = (drill.cursor(n)?, cursor(n % CYCLE)?);
        assert_eq!(
            drill.pending_prefix(&at)?,
            shuffled.pending_prefix(&local)?,
            "prefix at {n}"
        );
        assert_eq!(
            drill.anchor_target(&at)?,
            shuffled.anchor_target(&local)?,
            "target at {n}"
        );
    }
    for n in [0, 5, CYCLE + 1_280] {
        let prompt = drill.pending_prefix(&drill.cursor(n)?)?;
        assert!(prompt[48] == prompt[8] || prompt[48] == prompt[18]);
        let slot = if prompt[48] == prompt[8] { 12 } else { 22 };
        let answer: Vec<u8> = (n..n + 4)
            .map(|k| drill.anchor_target(&drill.cursor(k)?).map(|t| t.byte))
            .collect::<Result<_>>()?;
        assert_eq!(
            answer,
            prompt[slot..slot + 4],
            "drill answer must be the asked key's value"
        );
    }
    Ok(())
}

/// The corpus reads its pinned train file by window from disk: prefixes and targets equal the
/// in-memory window across the window edge and two epoch wraps; a tampered train file is refused.
#[test]
fn corpus_stream_reads_windows_and_wraps_epochs() -> Result<()> {
    let root = fresh_temp("corpus")?;
    std::fs::create_dir_all(&root)?;
    let train: Vec<u8> = (0..200u8).map(|i| b'a' + i % 26).collect();
    let heldout = b"a held-out story".to_vec();
    let entry = |bytes: &[u8]| json!({ "bytes": bytes.len(), "sha256": digest(bytes) });
    let manifest = serde_json::to_vec(&json!({ "schema": 1, "kind": "byte-corpus",
        "files": { "train.txt": entry(&train), "heldout.txt": entry(&heldout) } }))?;
    std::fs::write(root.join("train.txt"), &train)?;
    std::fs::write(root.join("heldout.txt"), &heldout)?;
    std::fs::write(root.join("manifest.json"), &manifest)?;
    let corpus = Corpus::open(&root, &digest(&manifest), 64)?;
    assert_eq!(window_before(&train, 0, 64), [START]);
    assert_eq!(
        window_before(&train, 63, 64),
        [&[START][..], &train[..63]].concat()
    );
    assert_eq!(window_before(&train, 64, 64), train[..64]);
    let mut cursor = corpus.cursor(0)?;
    for n in 0..450u64 {
        let t = (n % 200) as usize;
        assert_eq!((cursor.epoch, cursor.completion_offset), (n / 200, t));
        assert_eq!(
            corpus.pending_prefix(&cursor)?,
            window_before(&train, t, 64),
            "prefix at {n}"
        );
        assert_eq!(corpus.anchor_target(&cursor)?.byte, train[t]);
        corpus.advance_anchor(&mut cursor)?;
    }
    std::fs::write(root.join("train.txt"), &train[1..])?;
    assert!(Corpus::open(&root, &digest(&manifest), 64).is_err());
    std::fs::remove_dir_all(&root)?;
    Ok(())
}
