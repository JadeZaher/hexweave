//! Read-only context-reach measurement on an exact connected checkpoint; see CONNECTED.md
//! §Context reach.

use super::super::connected_state::State;
use super::super::core_config::{Config, Data};
use super::super::core_corpus::Corpus;
use super::super::geometry::Geometry;
use super::super::{connected_checkpoint, connected_eval, tuning};
use super::{fingerprint, io};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;

/// Whole-measurement deadline.
const TIMEOUT: Duration = Duration::from_secs(7200);

/// Held-out loss by retained context length for the checkpoint named in a run config.
pub(in super::super) fn reach_path(config: &Path, checkpoint: &str, spans: usize) -> Result<Value> {
    let config: Config = serde_json::from_slice(&std::fs::read(config)?)?;
    config.validate()?;
    reach(&config, checkpoint, spans, Geometry::candidate())
}

/// `reach_path` on a parsed config and explicit geometry.
pub(super) fn reach(
    config: &Config,
    checkpoint: &str,
    spans: usize,
    geometry: Geometry,
) -> Result<Value> {
    ensure!((2..=128).contains(&spans), "reach spans must be in 2..=128");
    let options = config
        .connected
        .context("context reach requires connected options")?;
    let Data::Corpus {
        root,
        manifest_sha256: manifest,
        window,
    } = &config.data
    else {
        anyhow::bail!("context reach requires a corpus checkpoint");
    };
    ensure!(
        (1..=geometry.context).contains(window),
        "context reach window exceeds context"
    );
    let mut scratch = io::Scratch::new()?;
    let source = io::checkpoint_path(&config.out, checkpoint)?;
    connected_checkpoint::check_tile_count(&source)?;
    let snapshot = scratch.path.join("checkpoints").join(checkpoint);
    let hash = io::snapshot(&source, &snapshot, geometry)?;
    let identity = config.identity()?;
    tuning::install(config.knobs());
    let workers = options.workers;
    let loaded =
        connected_checkpoint::load(&snapshot, &scratch.path, geometry, &identity, workers)?;
    let state = State::restore(&loaded.payloads, geometry, config.seed)?;
    ensure!(
        state.anchors == loaded.anchors
            && state.options.with_paging(&options) == options
            && loaded.store.precision == options.precision,
        "context reach state/options mismatch"
    );
    ensure!(
        &state.cursor.data.manifest_sha256 == manifest,
        "context reach corpus identity mismatch"
    );
    let corpus = Corpus::open(root, manifest, (*window).max(32))?;
    let mut store = loaded.store;
    let before = fingerprint(&state, &mut store)?;
    let mut report =
        connected_eval::context_reach(&state, &mut store, &corpus, spans, *window, TIMEOUT)?;
    ensure!(
        before == fingerprint(&state, &mut store)?
            && hash == io::checkpoint_hash(&snapshot, geometry)?,
        "context reach mutated loaded state or snapshot"
    );
    scratch.remove()?;
    report["checkpoint"] = json!(checkpoint);
    report["anchors"] = json!(state.anchors);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::super::contracts::fixture;
    use super::*;

    #[test]
    fn reach_ends_at_the_full_window_and_shorter_context_scores_differently() -> Result<()> {
        let fixture = fixture()?;
        let report = reach(&fixture.config, &fixture.name, 2, Geometry::fixture())?;
        let rows = report["rows"].as_array().context("reach rows")?;
        let (first, last) = (&rows[0], &rows[rows.len() - 1]);
        assert_eq!(first["context_bytes"], json!(1));
        assert_eq!(last["context_bytes"], json!(8));
        assert_ne!(first["bits_per_byte"], last["bits_per_byte"]);
        Ok(())
    }
}
