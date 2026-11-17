//! Read-only corpus checkpoint completion; see AGENTS.md §Core v2.

use super::core_config::{Config, Data};
use super::core_corpus::Corpus;
use super::data_contracts::Split;
use super::decoding::{story_opening, Decoder};
use super::geometry::Geometry;
use super::{core_checkpoint, generation, tuning};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::path::Path;

/// Generate greedily from one explicitly named, identity-checked corpus checkpoint.
pub(super) fn complete(
    config_path: &Path,
    checkpoint: &str,
    prompt: &str,
    limit: usize,
) -> Result<Value> {
    complete_options(
        config_path,
        checkpoint,
        prompt,
        limit,
        super::connected_decode::Options::default(),
        Decoder::Greedy,
    )
}

/// Continue `prompt` after a full window of held-out text; nucleus sampling is reference-path only.
pub(super) fn complete_options(
    config_path: &Path,
    checkpoint: &str,
    prompt: &str,
    limit: usize,
    decode: super::connected_decode::Options,
    decoder: Decoder,
) -> Result<Value> {
    decode.validate()?;
    decoder.validate()?;
    ensure!(
        checkpoint.len() == 13
            && checkpoint.starts_with('a')
            && checkpoint[1..].bytes().all(|b| b.is_ascii_digit()),
        "checkpoint must be a followed by 12 digits"
    );
    ensure!(
        !prompt.is_empty() && (1..=1023).contains(&limit),
        "prompt and byte limit must be nonempty and bounded"
    );
    ensure!(
        prompt
            .len()
            .checked_add(1)
            .is_some_and(|n| n <= Geometry::candidate().context),
        "prompt exceeds model context"
    );
    let config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    config.validate()?;
    if config.connected.is_some() {
        let decode = super::connected_decode::Options { decoder, ..decode };
        return super::connected_completion::complete(&config, checkpoint, prompt, limit, decode);
    }
    ensure!(
        decode.trace_bytes == 0 && decode.generated_feedback && !decode.quantize_audit,
        "decoder diagnostics require a connected checkpoint"
    );
    let (root, manifest, window) = match &config.data {
        Data::Corpus {
            root,
            manifest_sha256,
            window,
        } => (root, manifest_sha256, *window),
        _ => anyhow::bail!("completion requires a corpus checkpoint"),
    };
    let identity = config.identity()?;
    tuning::install(config.knobs());
    let path = config.out.join("checkpoints").join(checkpoint);
    let (state, mut store) = core_checkpoint::load(&path, Geometry::candidate(), &identity)?;
    ensure!(
        state.cursor.data.schema == "byte-corpus-v1"
            && &state.cursor.data.manifest_sha256 == manifest
            && state.cursor.data.dataset_version == format!("window-{window}")
            && state.cursor.data.split == Split::Train,
        "checkpoint corpus identity mismatch"
    );
    let heldout = Corpus::open(root, manifest, window)?.heldout;
    let opening = story_opening(&heldout, window, 0, 1, prompt.as_bytes())?;
    let result = generation::generate_story(
        &state,
        &mut store,
        &opening,
        limit,
        window,
        decoder,
        &mut || Ok(None),
    )?;
    Ok(json!({
        "checkpoint": checkpoint, "anchors": state.global_opportunity,
        "identity_sha256": identity, "generation_policy": "story-window-v2",
        "decoder": decoder, "context_bytes": opening.len() - prompt.len(),
        "text": String::from_utf8_lossy(&result.raw_bytes), "raw_bytes": result.raw_bytes,
        "valid_utf8": result.valid_utf8, "stop_reason": result.stop_reason,
        "before_sha256": result.before_sha256, "after_sha256": result.after_sha256,
        "immutability_scope": "packed-model-and-continuation",
    }))
}

#[cfg(test)]
#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;

/// Completion loads a verified checkpoint read-only, opens it after a full window of held-out text,
/// and samples reproducibly per seed.
#[cfg(test)]
#[test]
fn completion_uses_a_verified_checkpoint_without_mutating_it() -> Result<()> {
    use super::storage_paths::digest;
    use super::{core_config::Config, resident::ResidentStore, state::State};

    let corpus = super::scratch_dir::ScratchDir::new("completion-corpus")?;
    let heldout = [&[b'h'; 600][..], &[super::core_corpus::END], b"tail"].concat();
    let entry = |bytes: &[u8]| json!({ "bytes": bytes.len(), "sha256": digest(bytes) });
    let manifest_bytes = serde_json::to_vec(&json!({ "schema": 1, "kind": "byte-corpus",
        "files": { "train.txt": entry(b"train bytes"), "heldout.txt": entry(&heldout) } }))?;
    std::fs::write(corpus.join("train.txt"), b"train bytes")?;
    std::fs::write(corpus.join("heldout.txt"), &heldout)?;
    std::fs::write(corpus.join("manifest.json"), &manifest_bytes)?;
    let root = Path::new("work/core-runs");
    std::fs::create_dir_all(root)?;
    let out = root.join(format!("completion-contract-{}", std::process::id()));
    std::fs::create_dir(&out)?;
    let config_path = out.join("config.json");
    let manifest = digest(&manifest_bytes);
    let mut document = json!({ "schema": 1, "out": out, "data": {
        "kind": "corpus", "root": corpus.path(), "manifest_sha256": manifest,
        "window": 512 }, "seed": 7, "rate": 0.003, "anchors": 1, "wall_seconds": 1,
        "checkpoint_every": 1, "eval_every": 1, "eval_cases": 1, "resume": false });
    std::fs::write(&config_path, serde_json::to_vec(&document)?)?;
    let config: Config = serde_json::from_value(document.clone())?;
    let _guard = tuning::scoped(config.knobs());
    let source = fixture::Source::new();
    let mut cursor = source.cursor();
    cursor.data.schema = "byte-corpus-v1".into();
    cursor.data.manifest_sha256 = manifest;
    cursor.data.dataset_version = "window-512".into();
    let mut state = State::with_cursor(Geometry::candidate(), 7, cursor)?;
    state.anchor_only = true;
    let store = ResidentStore::new(state.geometry, 7)?;
    let checkpoint = core_checkpoint::save(&out, &state, &store, &config.identity()?)?;
    let before = std::fs::read(checkpoint.join("manifest.json"))?;
    let result = complete(&config_path, "a000000000000", "X", 3)?;
    assert_eq!(result["anchors"], 0);
    assert_eq!(result["before_sha256"], result["after_sha256"]);
    assert_eq!(
        result["context_bytes"], 511,
        "the prompt ends a full window"
    );
    assert_eq!(std::fs::read(checkpoint.join("manifest.json"))?, before);
    let sample = |top_p, seed| {
        let decoder = Decoder::Nucleus {
            temperature: 1.0,
            top_p,
            seed,
        };
        let options = super::connected_decode::Options::default();
        complete_options(&config_path, "a000000000000", "X", 8, options, decoder)
    };
    assert_eq!(sample(0.9, 3)?["raw_bytes"], sample(0.9, 3)?["raw_bytes"]);
    assert_ne!(sample(0.9, 3)?["raw_bytes"], sample(0.9, 4)?["raw_bytes"]);
    assert!(sample(0.0, 3).is_err());
    tuning::ripple_latents(&[((super::geometry::TILES, 0), &state.shared.embedding)])?;
    let (sentinel, _) = tuning::export();
    tuning::import(sentinel, 37);
    let before_optimizer = tuning::export();
    let before_knobs = tuning::current();
    super::complete(&config_path, "a000000000000", "X", 3)?;
    assert_eq!(tuning::export(), before_optimizer);
    assert!(super::complete(&config_path, "a000000000001", "X", 1).is_err());
    assert_eq!(tuning::export(), before_optimizer);
    assert_eq!(tuning::current(), before_knobs);
    assert!(complete(&config_path, "../escape", "X", 1).is_err());
    assert!(complete(&config_path, "a000000000000", "", 1).is_err());
    assert!(complete(&config_path, "a000000000000", "X", 1024).is_err());
    let missing = out.join("missing.json");
    let error = complete(&missing, "a000000000000", &"x".repeat(1024), 1).unwrap_err();
    assert!(error.to_string().contains("prompt exceeds model context"));
    document["seed"] = json!(8);
    std::fs::write(&config_path, serde_json::to_vec(&document)?)?;
    assert!(complete(&config_path, "a000000000000", "X", 1).is_err());
    assert!(checkpoint.exists());
    for entry in std::fs::read_dir(&checkpoint)? {
        std::fs::remove_file(entry?.path())?;
    }
    std::fs::remove_dir(&checkpoint)?;
    std::fs::remove_dir(out.join("checkpoints"))?;
    std::fs::remove_file(config_path)?;
    std::fs::remove_dir(out)?;
    Ok(())
}
