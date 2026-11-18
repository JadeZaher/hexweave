//! Read-only generation from an exact connected checkpoint; see CONNECTED.md.

use super::connected_state::State;
use super::connected_store::Store;
use super::core_config::{Config, Data};
use super::data_contracts::Split;
use super::geometry::Geometry;
use super::storage_paths::digest;
use super::{connected_checkpoint, connected_eval, tuning};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(test)]
#[path = "connected_completion_contracts.rs"]
mod contracts;
#[cfg(test)]
#[path = "execution_fixture.rs"]
#[allow(clippy::duplicate_mod, dead_code)]
mod fixture;
#[path = "connected_completion_io.rs"]
mod io;
#[cfg(all(test, feature = "tiles19"))]
#[path = "connected_tiles19_contracts.rs"]
mod tiles19;

const TIMEOUT: Duration = Duration::from_secs(120);

pub(super) fn complete(
    config: &Config,
    checkpoint: &str,
    prompt: &str,
    limit: usize,
    decode: super::connected_decode::Options,
) -> Result<Value> {
    complete_geometry_options(
        config,
        checkpoint,
        prompt,
        limit,
        Geometry::candidate(),
        decode,
    )
}

#[cfg(test)]
fn complete_geometry(
    config: &Config,
    checkpoint: &str,
    prompt: &str,
    limit: usize,
    geometry: Geometry,
) -> Result<Value> {
    complete_geometry_options(
        config,
        checkpoint,
        prompt,
        limit,
        geometry,
        super::connected_decode::Options::default(),
    )
}

fn complete_geometry_options(
    config: &Config,
    checkpoint: &str,
    prompt: &str,
    limit: usize,
    geometry: Geometry,
    decode: super::connected_decode::Options,
) -> Result<Value> {
    decode.validate()?;
    ensure!(
        !decode.quantize_audit
            || config
                .connected
                .is_some_and(|o| o.precision == super::connected_precision::Precision::Fp32),
        "quantize-audit requires a connected FP32 checkpoint"
    );
    let owner_lock_present = config.out.join("OWNER.lock").exists();
    ensure!(
        !prompt.is_empty()
            && prompt
                .len()
                .checked_add(1)
                .is_some_and(|len| len <= geometry.context)
            && (1..=1023).contains(&limit),
        "invalid connected completion prompt or byte limit"
    );
    let started = Instant::now();
    let mut scratch = io::Scratch::new()?;
    let mut result = infer(
        config,
        checkpoint,
        prompt,
        limit,
        geometry,
        &scratch.path,
        started,
        decode,
    )?;
    scratch.remove()?;
    result["scratch_removed"] = json!(true);
    result["owner_lock_present"] = json!(owner_lock_present);
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn infer(
    config: &Config,
    checkpoint: &str,
    prompt: &str,
    limit: usize,
    geometry: Geometry,
    scratch: &Path,
    started: Instant,
    decode: super::connected_decode::Options,
) -> Result<Value> {
    let source = io::checkpoint_path(&config.out, checkpoint)?;
    // Another build's checkpoint lacks this build's tile files; name the build instead.
    connected_checkpoint::check_tile_count(&source)?;
    // Read a possibly live source once; the hash then covers exactly the bytes loaded.
    let snapshot = scratch.join("checkpoints").join(checkpoint);
    let hash = io::snapshot(&source, &snapshot, geometry)?;
    infer_snapshot(
        config, &snapshot, &hash, prompt, limit, geometry, scratch, started, decode,
    )
}

#[allow(clippy::too_many_arguments)]
fn infer_snapshot(
    config: &Config,
    path: &Path,
    checkpoint_before: &str,
    prompt: &str,
    limit: usize,
    geometry: Geometry,
    scratch: &Path,
    started: Instant,
    decode: super::connected_decode::Options,
) -> Result<Value> {
    let checkpoint = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("invalid connected snapshot name")?;
    let options = config
        .connected
        .context("connected completion requires connected options")?;
    let (root, manifest, window) = match &config.data {
        Data::Corpus {
            root,
            manifest_sha256,
            window,
        } => (root, manifest_sha256, *window),
        _ => anyhow::bail!("connected completion requires a corpus checkpoint"),
    };
    ensure!(
        (1..=geometry.context).contains(&window),
        "connected completion window exceeds context"
    );
    let identity = config.identity()?;
    tuning::install(config.knobs());
    let loaded = connected_checkpoint::load(path, scratch, geometry, &identity, options.workers)?;
    let state = State::restore(&loaded.payloads, geometry, config.seed)?;
    ensure!(
        state.anchors == loaded.anchors
            && state.options.with_paging(&options) == options
            && loaded.store.precision == options.precision,
        "connected completion state/options mismatch"
    );
    ensure!(
        state.cursor.data.schema == "byte-corpus-v1"
            && &state.cursor.data.manifest_sha256 == manifest
            && state.cursor.data.dataset_version == format!("window-{window}")
            && state.cursor.data.split == Split::Train,
        "connected checkpoint corpus identity mismatch"
    );
    // Only the verified held-out bytes are read; `open`'s window floor (32) does not affect them.
    let heldout = super::core_corpus::Corpus::open(root, manifest, window.max(32))?.heldout;
    let opening = super::decoding::story_opening(&heldout, window, 0, 1, prompt.as_bytes())?;
    let context = &opening[..opening.len() - prompt.len()];
    let mut store = loaded.store;
    let before = fingerprint(&state, &mut store)?;
    let generation = generate(
        &state, &mut store, context, prompt, window, limit, started, decode,
    )?;
    let after = fingerprint(&state, &mut store)?;
    let checkpoint_after = io::checkpoint_hash(path, geometry)?;
    ensure!(
        before == after && checkpoint_before == checkpoint_after,
        "connected inference mutated loaded state or snapshot"
    );
    let mut report = generation;
    let fields = json!({
        "checkpoint": checkpoint, "anchors": state.anchors, "identity_sha256": identity,
        "protocol": options.precision.protocol(), "precision": options.precision,
        "shared_view_bytes": state.shared.weights.as_ref().map_or(0, |weights| weights.bytes()),
        "peak_shared_view_bytes": state.shared.weights.as_ref().map_or(0, |weights| weights.bytes() * 2),
        "generation_policy": super::connected_decode::POLICY,
        "before_sha256": before, "after_sha256": after,
        "checkpoint_before_sha256": checkpoint_before, "checkpoint_after_sha256": checkpoint_after,
        "immutability_scope": "source-training-state-packed-tiles-Adam-and-trit-direction-history",
        "excluded_from_fingerprint": ["read-and-cache-statistics", "isolated-generation-memory"],
        "weights_and_predictor_frozen": true, "training_stop_ignored": true,
        "budget_seconds": TIMEOUT.as_secs(), "elapsed_seconds": started.elapsed().as_secs_f64(),
        "paging": store.stats(), "corpus_opened": true, "context_bytes": context.len(),
    });
    report
        .as_object_mut()
        .context("invalid generation response")?
        .extend(
            fields
                .as_object()
                .context("invalid completion metadata")?
                .clone(),
        );
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn generate(
    state: &State,
    store: &mut Store,
    context: &[u8],
    prompt: &str,
    window: usize,
    limit: usize,
    started: Instant,
    decode: super::connected_decode::Options,
) -> Result<Value> {
    if decode.quantize_audit {
        super::connected_quantization::compare(
            state, store, context, prompt, window, limit, started, TIMEOUT, decode,
        )
    } else {
        connected_eval::generate_frozen_with_options(
            state, store, context, prompt, window, limit, started, TIMEOUT, decode,
        )
    }
}

fn fingerprint(state: &State, store: &mut Store) -> Result<String> {
    let shared = state
        .payloads()?
        .into_iter()
        .map(|(name, bytes)| (name, digest(&bytes)))
        .collect::<Vec<_>>();
    Ok(digest(&serde_json::to_vec(&(
        shared,
        store.fingerprint()?,
    ))?))
}
