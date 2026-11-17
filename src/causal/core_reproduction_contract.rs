//! Opt-in read-only reproduction of the recorded v1 public-corpus checkpoint.

use super::checkpoint_state::StateRecord;
use super::core_codec::{decode_latents, decode_tiles};
use super::core_corpus::Corpus;
use super::state_codec::{decode_predictor, decode_shared};
use super::storage_paths::digest;
use super::tuning::{self, Knobs};
use anyhow::{ensure, Result};
use serde_json::Value;
use std::path::Path;

/// Reproduce the handoff's logged score without migration or training; see AGENTS.md §Core v2.
#[test]
#[ignore = "requires the owner's saved public TinyStories checkpoint under work/"]
fn saved_v1_evaluation_reproduces_logged_bits() -> Result<()> {
    let run = Path::new("work/core-runs/rescore-abl-on-100k-16");
    let checkpoint = run.join("checkpoints/a000000100000");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("manifest.json"))?)?;
    ensure!(
        manifest["protocol"] == "causal-latent-v1",
        "expected the recorded v1 checkpoint"
    );
    let read = |name: &str| -> Result<Vec<u8>> {
        let bytes = std::fs::read(checkpoint.join(name))?;
        ensure!(
            manifest["files"][name] == digest(&bytes),
            "checkpoint hash mismatch: {name}"
        );
        Ok(bytes)
    };
    let _guard = tuning::scoped(Knobs::core(0.003, 7));
    let record: StateRecord = serde_json::from_slice(&read("state.json")?)?;
    let geometry = record.geometry;
    let shared = decode_shared(&read("shared.bin")?, geometry, record.shared_version)?;
    let predictor = decode_predictor(&read("predictor.bin")?, geometry, record.predictor_version)?;
    let mut store = decode_tiles(&read("tiles.bin")?, geometry, record.tile_versions)?;
    let state = record.restore(shared, predictor)?;
    tuning::import(
        decode_latents(&read("latent.bin")?)?,
        manifest["calls"].as_u64().unwrap(),
    );
    let corpus = Corpus::open(
        Path::new("work/corpora/tinystories"),
        "2710d21277ed6e1817477ae6a61444ace674b7dc371a1ec3ea304a9e93575af8",
        512,
    )?;
    let report: Value = serde_json::from_slice(&std::fs::read(run.join("report.json"))?)?;
    let actual = super::core_eval::corpus(&state, &mut store, &corpus, 16)?;
    for key in ["dev_bits_per_byte", "alternate_bits_per_byte"] {
        let expected = report["final"]["eval"][key].as_f64().unwrap();
        let got = actual[key].as_f64().unwrap();
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "{key}: {got:?} vs {expected:?}"
        );
        println!("{key}: {got:?} (exact reproduction)");
    }
    Ok(())
}

/// Compare greedy generation with and without replay on the saved public v2 state.
#[test]
#[ignore = "requires the completed public v2 scan checkpoint under work/"]
fn saved_v2_greedy_history_diagnostic() -> Result<()> {
    use super::core_config::Config;
    use super::geometry::Geometry;

    let run = Path::new("work/core-runs/tiny-v2-b32-s7-scan-r003-20261102");
    let config: Config = serde_json::from_slice(&std::fs::read(run.with_extension("json"))?)?;
    config.validate()?;
    let knobs = config.knobs();
    let _guard = tuning::scoped(knobs);
    let preserved =
        Path::new("work/core-runs/scan-final-preserved").join("tiny-v2-b32-s7-scan-r003-20261102");
    let checkpoint = preserved.join("a000000392640");
    let (mut state, mut store) =
        super::core_checkpoint::load(&checkpoint, Geometry::candidate(), &config.identity()?)?;
    let retained = state.history.clone();
    let report: Value = serde_json::from_slice(&std::fs::read(preserved.join("report.json"))?)?;
    let prompts = [
        "Once upon a time",
        "One day, a little girl named Lily",
        "Tom and his dog went to the",
        "The sun was",
    ];
    for history in [true, false] {
        state.history = if history {
            retained.clone()
        } else {
            Default::default()
        };
        tuning::anneal(Knobs { history, ..knobs });
        for (index, prompt) in prompts.iter().enumerate() {
            let opening = [&[super::core_corpus::END][..], prompt.as_bytes()].concat();
            let result = super::generation::generate_story(
                &state,
                &mut store,
                &opening,
                160,
                512,
                super::decoding::Decoder::Greedy,
                &mut || Ok(None),
            )?;
            if history {
                let expected = report["final"]["eval"]["samples"][index]["text"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("missing saved sample"))?;
                ensure!(
                    String::from_utf8_lossy(&result.raw_bytes) == expected,
                    "saved generation mismatch"
                );
            }
            println!(
                "{}",
                serde_json::json!({
                    "history": history, "prompt": prompt,
                    "text": String::from_utf8_lossy(&result.raw_bytes),
                    "valid_utf8": result.valid_utf8, "stop_reason": result.stop_reason,
                    "before_sha256": result.before_sha256, "after_sha256": result.after_sha256,
                })
            );
        }
    }
    tuning::anneal(knobs);
    state.history = retained;
    state.validate_boundary()?;
    Ok(())
}
