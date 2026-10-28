//! Core-path contracts: harness parity, drill stream identity, resume and fallback; see AGENTS.md §core.

use super::assay_observation;
use super::assay_source::{cursor, Source};
use super::assay_state::{Progress, MANIFEST_SHA};
use super::assay_update::Rule;
use super::checkpoint_owner::Terminal;
use super::core_checkpoint;
use super::core_codec::encode_latents;
use super::core_drill::{Drill, CYCLE};
use super::execution::{self, TrainingSource};
use super::geometry::Geometry;
use super::resident::ResidentStore;
use super::state::State;
use super::state_codec::{encode_predictor, encode_shared, encode_tile};
use super::tuning::{self, Knobs};
use super::tuning_sources::Shuffled;
use anyhow::{Context, Result};
use std::path::Path;

fn assay() -> Result<Source> {
    Source::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/retained-text-assay"),
        MANIFEST_SHA,
    )
}

/// Everything training numerics touch: tiles, E, predictor, latents, history, baseline, loss.
fn fingerprint(state: &State, store: &ResidentStore) -> Result<Vec<Vec<u8>>> {
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

fn train<S: TrainingSource>(
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
    let out = std::env::temp_dir().join(format!("hexweave-parity-{}", std::process::id()));
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

/// Resume is bit-exact; a corrupt newest checkpoint is set aside for the previous one, a stale
/// partial is ignored, and another run identity is refused without touching anything.
#[test]
fn core_resume_is_bit_exact_and_falls_back_past_corruption() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    let geometry = Geometry::candidate();
    let out = std::env::temp_dir().join(format!("hexweave-core-{}", std::process::id()));
    let _core = tuning::scoped(Knobs::core(0.003, 7));
    let mut state = State::with_cursor(geometry, 7, drill.cursor(0)?)?;
    state.anchor_only = true;
    let mut store = ResidentStore::new(geometry, 7)?;
    train(&mut state, &mut store, &drill, 5)?;
    core_checkpoint::save(&out, &state, &store, "identity")?;
    train(&mut state, &mut store, &drill, 5)?;
    let newest = core_checkpoint::save(&out, &state, &store, "identity")?;
    train(&mut state, &mut store, &drill, 5)?;
    let uninterrupted = fingerprint(&state, &store)?;
    let tiles = newest.join("tiles.bin");
    let mut bytes = std::fs::read(&tiles)?;
    bytes[100] ^= 1;
    std::fs::write(&tiles, bytes)?;
    std::fs::create_dir_all(out.join("checkpoints").join("a000000000015.partial"))?;
    assert!(core_checkpoint::resume(&out, geometry, "other").is_err());
    assert!(
        newest.exists(),
        "a foreign identity must not quarantine checkpoints"
    );
    tuning::install(Knobs::core(0.003, 7));
    let (mut resumed, mut store, path) =
        core_checkpoint::resume(&out, geometry, "identity")?.context("no checkpoint resumed")?;
    assert!(path.ends_with("a000000000005") && !newest.exists());
    train(&mut resumed, &mut store, &drill, 10)?;
    assert!(
        fingerprint(&resumed, &store)? == uninterrupted,
        "fallback resume must be bit-exact"
    );
    std::fs::remove_dir_all(&out)?;
    Ok(())
}
