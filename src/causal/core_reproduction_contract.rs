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
