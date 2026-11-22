//! Contextual-row admission, embedding-gradient oracle and train/evaluate/resume flow; see
//! CONNECTED.md §Contextual rows.

use super::super::connected_config::{self, CreditMode};
use super::super::connected_context;
use super::super::connected_route::RouteTraining;
use super::super::connected_step;
use super::super::core_config::Config;
use super::super::core_corpus::window_before;
use super::super::execution::TrainingSource;
use super::super::geometry::{Geometry, TILES};
use super::super::scratch_dir::ScratchDir;
use super::super::storage_paths::digest;
use super::super::tuning::{self, Knobs};
use super::*;

/// Twenty bytes of train and held-out text: 24 anchors wrap the stream once at anchor 20.
const TEXT: &[u8; 20] = b"the cat sat. a dog.\x03";

fn corpus(path: &Path) -> Result<Corpus> {
    let entry = json!({"bytes":TEXT.len(),"sha256":digest(TEXT)});
    let manifest = serde_json::to_vec(&json!({"schema":1,"kind":"byte-corpus",
        "files":{"train.txt":entry,"heldout.txt":entry}}))?;
    std::fs::write(path.join("train.txt"), TEXT)?;
    std::fs::write(path.join("heldout.txt"), TEXT)?;
    std::fs::write(path.join("manifest.json"), &manifest)?;
    let mut corpus = Corpus::open(path, &digest(&manifest), 32)?;
    corpus.window = Geometry::fixture().context;
    Ok(corpus)
}

/// Greedy routes and zero history, so training and frozen replay read identical tile inputs.
fn options() -> connected_config::Options {
    connected_config::Options {
        lanes: 2,
        workers: 2,
        visits: 2,
        credit_mode: CreditMode::AnchorExact,
        route_training: RouteTraining::Greedy,
        history: false,
        context_rows: true,
        ..connected_config::Options::default()
    }
}

#[test]
fn context_rows_are_identity_under_anchor_exact_and_refuse_other_row_readers() -> Result<()> {
    let base = json!({"schema":1,"out":"work/core-runs/context-test",
        "data":{"kind":"corpus","root":"work/corpora/test","manifest_sha256":"test","window":32},
        "seed":7,"rate":0.003,"anchors":8,"wall_seconds":60,"checkpoint_every":8,"eval_every":8,
        "eval_cases":1,"resume":false,"connected":{"credit_mode":"anchor_exact"},"tiles":TILES});
    let identity = serde_json::from_value::<Config>(base.clone())?.identity()?;
    // (extra key, value, admitted): waves, sections and both replays are refused by this switch.
    for (key, value, admitted) in [
        ("context_rows", json!(true), true),
        ("anchor_batch", json!(2), false),
        ("shared", json!({"sections":1}), false),
        ("replay", json!({"capacity":64,"per_anchor":2}), false),
        (
            "restart",
            json!({"floor":0.1,"threshold":0.5,"replay":2}),
            false,
        ),
    ] {
        let mut changed = base.clone();
        changed["connected"]["context_rows"] = json!(true);
        changed["connected"][key] = value;
        let config: Config = serde_json::from_value(changed)?;
        match config.validate() {
            Ok(()) => assert!(admitted && config.identity()? != identity, "{key}"),
            Err(error) => assert!(!admitted && format!("{error:#}").contains("context_rows")),
        }
    }
    Ok(())
}

#[test]
fn frozen_replay_resume_and_generation_reproduce_training_rows() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let path = ScratchDir::new("connected-context")?;
    let corpus = corpus(&path)?;
    let (geometry, options) = (Geometry::fixture(), options());
    let mut state = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let mut store = Store::new(&path.join("tiles"), geometry, 7, 2)?;
    for _ in 0..24 {
        connected_step::step(&mut state, &mut store, &corpus, options, 0.003, false)?;
    }
    // The stream restarted at anchor 20: nothing older survives, though capacity 8 would keep 19.
    let held = |state: &State, key| Some(state.context.as_ref()?.state(key)?.to_vec());
    assert!(held(&state, 19).is_none() && held(&state, 20).is_some());
    // (a) Anchor 24 = `[START] t h e c`: older rows carry states 20..23, the newest stays raw.
    let bytes = corpus.pending_prefix(&state.cursor)?;
    let buffer = state
        .context
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("buffer"))?;
    let (rows, tally) = connected_context::rows(&state.shared, &bytes, buffer, 24, 0)?;
    let raw = state.shared.prefix(&bytes)?;
    let d = geometry.width;
    assert_eq!((bytes.len(), tally.filled), (5, 4));
    for index in 0..5 {
        let range = index * d..(index + 1) * d;
        assert_eq!(rows.memory[range.clone()] == raw.memory[range], index == 4);
    }
    // (d) A resumed copy refills positions 20..23 under the same weights.
    let mut resumed = State::restore(&state.payloads()?, geometry, 7)?;
    connected_context::refill(&mut resumed, &mut store, &corpus)?;
    let refilled: Vec<_> = (20..24).map(|key| held(&resumed, key)).collect();
    assert!(refilled.iter().all(Option::is_some) && held(&resumed, 19).is_none());
    // Training-order forward passes (frozen: no update) from offset 4 through 9.
    let mut training = Vec::new();
    for _ in 4..10 {
        let revisions = std::array::from_fn(|id| store.slots[id].meta.revision);
        let (_, _, trace) =
            connected_step::forward(&mut resumed, &mut store, &corpus, options, &revisions)?;
        let target = corpus.anchor_target(&resumed.cursor)?.byte;
        training.push(resumed.shared.readout(&trace.output)?.loss(target)?);
        corpus.advance_anchor(&mut resumed.cursor)?;
        resumed.anchors += 1;
    }
    // (b) Frozen evaluation warms its own buffer to the refill's states and then scores
    // exactly the training-order losses.
    let before = state.payloads()?;
    let mut replay = replay::Replay::new(&state, &mut store)?;
    assert_eq!(probe::observed_from(&replay, TEXT, 6, 8), 0);
    for t in 0..10 {
        let scored = replay.score_at(&window_before(TEXT, t, 8), TEXT[t], t as u64, 0)?;
        if t == 3 {
            let warmed: Vec<_> = (0..4).map(|key| replay.buffered(key)).collect();
            assert_eq!(warmed, refilled);
        }
        assert!(t < 4 || scored == training[t - 4], "position {t}");
    }
    drop(replay);
    refill_uses_the_live_anchor_draw(&state, &mut store, &corpus)?;
    generation_and_measurement_read_the_fill(&state, &mut store, &corpus, &path)?;
    assert_eq!(before, state.payloads()?);
    store.quiescent()?;
    drop(store);
    drop(corpus);
    Ok(())
}

/// (c) Generation after a full window is deterministic and reads its warm-up; (e) evaluation and
/// context reach (whose full-window row must equal held-out) run on the frozen state.
fn generation_and_measurement_read_the_fill(
    state: &State,
    store: &mut Store,
    corpus: &Corpus,
    path: &Path,
) -> Result<()> {
    let decode = super::super::connected_decode::Options {
        trace_bytes: 1,
        ..Default::default()
    };
    let budget = Budget {
        started: Instant::now(),
        until: Duration::from_secs(60),
        out: None,
    };
    let mut generate = |warm: bool| -> Result<Value> {
        let replay = replay::Replay::new(state, store)?;
        let mut replay = if warm {
            replay
        } else {
            replay.without_warm_up()
        };
        replay.generate(&TEXT[..8], "a d", 8, 3, &budget, decode)
    };
    let (first, again, cold) = (generate(true)?, generate(true)?, generate(false)?);
    assert_eq!(first, again);
    let logits = |value: &Value| value["decode_trace"][0]["logits"].clone();
    assert_ne!(logits(&first), logits(&cold));
    let assessment = super::Options {
        budget_seconds: 60,
        retention_spans: 0,
        sample_bytes: 2,
    };
    evaluate(state, store, corpus, 2, path, 1, assessment, false)?;
    let until = Duration::from_secs(60);
    context_reach(state, store, corpus, 2, corpus.window, until)?;
    Ok(())
}

/// Under sampled routes, a live forward at anchor 24 and a refill ending at 24 (same weights)
/// push the same state: the refill takes the live anchor's salted route draw.
fn refill_uses_the_live_anchor_draw(
    state: &State,
    store: &mut Store,
    corpus: &Corpus,
) -> Result<()> {
    let sampled = |anchors: u64| -> Result<State> {
        let mut copy = State::restore(&state.payloads()?, state.geometry, 7)?;
        copy.options.route_training = RouteTraining::Sampled;
        (copy.cursor, copy.anchors) = (corpus.cursor(anchors)?, anchors);
        Ok(copy)
    };
    let (mut live, mut refilled) = (sampled(24)?, sampled(25)?);
    connected_context::refill(&mut live, store, corpus)?;
    let options = live.options;
    let revisions = std::array::from_fn(|id| store.slots[id].meta.revision);
    connected_step::forward(&mut live, store, corpus, options, &revisions)?;
    connected_context::refill(&mut refilled, store, corpus)?;
    let state_at = |copy: &State| {
        copy.context
            .as_ref()
            .and_then(|c| c.state(24).map(<[f32]>::to_vec))
    };
    assert!(state_at(&live).is_some());
    assert_eq!(state_at(&live), state_at(&refilled));
    Ok(())
}

#[test]
fn warm_up_observes_true_context_like_a_continuous_replay() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let path = ScratchDir::new("connected-context-warm")?;
    let corpus = corpus(&path)?;
    let options = connected_config::Options {
        history: true,
        ..options()
    };
    let state = State::new(Geometry::fixture(), 7, corpus.cursor(0)?, &options)?;
    let mut store = Store::new(&path.join("tiles"), Geometry::fixture(), 7, 2)?;
    let budget = Budget {
        started: Instant::now(),
        until: Duration::from_secs(60),
        out: None,
    };
    // Counted from 12: observed from 3, a full window before position 10 (two before 12).
    let probes = [probe::span(TEXT, 12, 6)];
    let scored = probe::score(&state, &mut store, &probes, 8, &budget)?;
    let mut sums = Vec::new();
    for (from, warm) in [(3, true), (10, false)] {
        let replay = replay::Replay::new(&state, &mut store)?;
        let mut replay = if warm {
            replay
        } else {
            replay.without_warm_up()
        };
        let mut sum = 0.0;
        for (t, &target) in TEXT.iter().enumerate().take(18).skip(from) {
            let value = replay.score_at(&window_before(TEXT, t, 8), target, t as u64, 0)?;
            sum += if t >= 12 { f64::from(value) } else { 0.0 };
        }
        sums.push(sum);
    }
    // Equal to a replay observing every position from 3 (memory and rows), unlike one from 10.
    assert_eq!(scored["probes"][0]["nll_sum"].as_f64(), Some(sums[0]));
    assert_ne!(sums[0], sums[1]);
    drop(store);
    drop(corpus);
    Ok(())
}
