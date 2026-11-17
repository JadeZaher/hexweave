//! Fresh-run readout prior and baseline, the readout-bias master's persistence, and which
//! learner each training rate drives (v5.1 `local_rate`, v5.2 `local_decay`).

use super::connected_checkpoint as checkpoint;
use super::connected_config::Options;
use super::connected_predictor::Predictor;
use super::connected_state::State;
use super::connected_step;
use super::connected_storage_contracts::fresh;
use super::connected_store::Store;
use super::connected_train::{fresh_state, readout_prior};
use super::connected_update::ROUTE_P_SLOT;
use super::core_corpus::Corpus;
use super::core_schedule::Decay;
use super::geometry::{Geometry, TILES, VOCAB};
use super::storage_paths::digest;
use super::tile_gradients::ROUTER;
use super::tuning::{self, Knobs, Latent, Latents};
use anyhow::{Context, Result};
use serde_json::json;
use std::f64::consts::FRAC_1_SQRT_2;
use std::path::Path;

/// A pinned byte corpus over `train` in `root`, its window cut to the fixture context.
pub(super) fn corpus(root: &Path, train: &[u8]) -> Result<Corpus> {
    let heldout = b"A small story.";
    let entry = |bytes: &[u8]| json!({"bytes":bytes.len(),"sha256":digest(bytes)});
    let manifest = serde_json::to_vec(&json!({"schema":1,"kind":"byte-corpus",
        "files":{"train.txt":entry(train),"heldout.txt":entry(heldout)}}))?;
    std::fs::create_dir_all(root)?;
    std::fs::write(root.join("train.txt"), train)?;
    std::fs::write(root.join("heldout.txt"), heldout)?;
    std::fs::write(root.join("manifest.json"), &manifest)?;
    let mut corpus = Corpus::open(root, &digest(&manifest), 32)?;
    corpus.window = Geometry::fixture().context;
    Ok(corpus)
}

#[test]
fn fresh_state_trains_from_the_corpus_prior_round_trips_and_enters_the_common_hash() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let options = Options {
        lanes: 2,
        workers: 2,
        visits: 2,
        ..Options::default()
    };
    let (out, geometry) = (fresh("bias")?, Geometry::fixture());
    let train: Vec<u8> = b"abacabad".iter().cycle().take(512).copied().collect();
    let corpus = corpus(&out.join("corpus"), &train)?;
    // The production seam: a normalized corpus log-unigram, its entropy the route baseline.
    let mut state = fresh_state(geometry, 7, &corpus, &options)?;
    let prior = readout_prior(corpus.byte_counts());
    assert!((prior.iter().map(|p| f64::from(*p).exp()).sum::<f64>() - 1.0).abs() < 1e-5);
    assert_eq!(state.shared.bias.as_ref(), Some(&prior));
    let entropy: f64 = prior
        .iter()
        .map(|&p| -f64::from(p).exp() * f64::from(p))
        .sum();
    assert!((f64::from(state.baseline) - entropy).abs() < 1e-5 && entropy < 3.0);
    assert!(state.anchors == 0 && state.bias_latent.is_none());
    state.validate()?;
    let mut store = Store::new(&out.join("connected-tiles"), geometry, 7, 2)?;
    connected_step::step(&mut state, &mut store, &corpus, options, 0.003, false)?;
    // One Adam step from the prior (a zero-started master would sit nats away); target 'a' rises.
    let master = state.bias_latent.clone().context("no bias master")?.weights;
    assert!((0..VOCAB).all(|b| (master[b] - prior[b]).abs() <= 0.01));
    assert!(master[usize::from(b'a')] > prior[usize::from(b'a')]);
    // The step recorded the raw visit scale: the anchor's frame mean, under the run maximum.
    let counters = &state.counters;
    assert!(counters.visit_rms_sum > 0.0 && counters.visit_rms_max >= counters.visit_rms_sum);
    // Once training has begun, by the anchor clock or the bias master, a prior is refused.
    let mut clocked = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    let mut mastered = State::new(geometry, 7, corpus.cursor(0)?, &options)?;
    clocked.anchors = 1;
    mastered.bias_latent = state.bias_latent.clone();
    for started in [&mut state, &mut clocked, &mut mastered] {
        assert!(started.install_readout_prior(&prior).is_err());
    }
    let bias = state.shared.bias.clone().context("missing readout bias")?;
    let payloads = state.payloads()?;
    checkpoint::save(&out, &mut store, "identity", state.anchors, &payloads)?;
    drop(store);
    let loaded = checkpoint::resume(&out, geometry, "identity", 2)?.context("no checkpoint")?;
    let mut restored = State::restore(&loaded.payloads, geometry, 7)?;
    assert_eq!(restored.shared.bias, Some(bias));
    assert_eq!(restored.bias_latent, state.bias_latent);
    let before = digest(&restored.payloads()?["connected.json"]);
    // One bias entry, moved together with its Adam master so the state stays consistent.
    let byte = usize::from(b'e');
    restored.shared.bias.as_mut().context("missing bias")?[byte] += 0.25;
    let master = restored.bias_latent.as_mut().context("missing master")?;
    master.weights[byte] += 0.25;
    assert_ne!(digest(&restored.payloads()?["connected.json"]), before);
    // Splitting the bias from its master, dropping the master or running its clock is refused.
    let original = restored.shared.bias.as_ref().context("missing bias")?[byte];
    restored.shared.bias.as_mut().context("missing bias")?[byte] += 0.5;
    assert!(restored.validate().is_err(), "bias diverged");
    // Restore the exact value: `(x + 0.5) - 0.5` need not round-trip in f32.
    restored.shared.bias.as_mut().context("missing bias")?[byte] = original;
    restored.validate()?;
    let latent = restored.bias_latent.take();
    assert!(restored.validate().is_err(), "no master");
    restored.bias_latent = latent;
    restored.bias_latent.as_mut().context("no master")?.step += 1;
    assert!(restored.validate().is_err(), "clock ahead");
    Ok(())
}

/// One arm's measured step: its applied local norm and the learners local steps must not touch.
struct Arm {
    applied: f64,
    routes: Latents,
    embedding: Option<Latent>,
    bias: Option<Latent>,
    predictor: Predictor,
}

/// Eight anchors at `base`, then anchor 8 alone at `measured`, in its own store.
fn arm(out: &Path, corpus: &Corpus, base: Options, name: &str, measured: Options) -> Result<Arm> {
    let geometry = Geometry::fixture();
    let mut state = State::new(geometry, 7, corpus.cursor(0)?, &base)?;
    let mut store = Store::new(&out.join(name), geometry, 7, 2)?;
    for _ in 0..8 {
        connected_step::step(&mut state, &mut store, corpus, base, 0.003, false)?;
    }
    state.options = measured;
    let before = state.counters.clone();
    connected_step::step(&mut state, &mut store, corpus, measured, 0.003, false)?;
    assert!(
        state.counters.local_updates > before.local_updates,
        "{name}: no local step"
    );
    // Route masters: ROUTER and P's route rows, written by route steps and never by local steps.
    let mut routes = Vec::new();
    for tile in 0..TILES {
        let lease = store.checkout(tile)?;
        let latents = lease.latents().iter();
        routes.extend(
            latents
                .filter(|(key, _)| key.1 == ROUTER || key.1 == ROUTE_P_SLOT)
                .cloned(),
        );
        store.return_read(lease)?;
    }
    Ok(Arm {
        applied: state.counters.local_applied_norm_sum - before.local_applied_norm_sum,
        routes,
        embedding: state.shared_latent,
        bias: state.bias_latent,
        predictor: state.predictor,
    })
}

#[test]
fn only_predicted_credit_tile_steps_follow_local_rate_and_decay() -> Result<()> {
    let _guard = tuning::scoped(Knobs {
        connected: true,
        ..Knobs::core(0.003, 7)
    });
    let base = Options {
        lanes: 2,
        workers: 2,
        visits: 2,
        local_rate: 1e-5,
        ..Options::default()
    };
    let out = fresh("local-rate")?;
    let corpus = corpus(&out.join("corpus"), &[b'a'; 256])?;
    let slow = arm(&out, &corpus, base, "slow", base)?;
    assert!(!slow.routes.is_empty(), "no route master to compare");
    // Anchor 8 at 10x the local rate, or under inverse square-root decay over 8 anchors (1/sqrt 2).
    let fast = Options {
        local_rate: 10.0 * base.local_rate,
        ..base
    };
    let decaying = Options {
        local_decay: Decay::InverseSqrt { anchors: 8 },
        ..base
    };
    for (name, measured, expected) in [("fast", fast, 10.0), ("decaying", decaying, FRAC_1_SQRT_2)]
    {
        let other = arm(&out, &corpus, base, name, measured)?;
        // Adam's step is linear in its rate, and the arms share moments and realized multipliers.
        let ratio = other.applied / slow.applied;
        assert!(
            (ratio / expected - 1.0).abs() < 1e-5,
            "{name}: applied local norm ratio {ratio}"
        );
        // Routes, E, the readout bias and the predictor step before the lanes, at their own rates.
        assert_eq!(slow.routes, other.routes, "{name}");
        assert_eq!(slow.embedding, other.embedding, "{name}");
        assert_eq!(slow.bias, other.bias, "{name}");
        assert_eq!(slow.predictor, other.predictor, "{name}");
    }
    Ok(())
}
