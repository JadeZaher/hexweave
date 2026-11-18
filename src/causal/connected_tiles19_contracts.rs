//! Nineteen-tile (`tiles19`) paged train, resume and completion flow; see CONNECTED.md §Tile count.

use super::super::connected_checkpoint as checkpoint;
use super::super::connected_config::Options;
use super::super::connected_precision::Precision;
use super::super::connected_state::State;
use super::super::connected_step;
use super::super::connected_store::Store;
use super::super::core_config::Config;
use super::super::core_corpus::Corpus;
use super::super::geometry::{Geometry, TILES};
use super::super::storage_paths::digest;
use super::super::tuning;
use super::{complete_geometry, io};
use anyhow::Result;
use serde_json::json;
use std::path::Path;

const RESIDENT: usize = 4;

/// A pinned corpus cut to the fixture window, with the run config completion reads it through.
fn corpus(root: &Path) -> Result<(Corpus, Config, Options)> {
    let corpus_root = root.join("corpus");
    std::fs::create_dir(&corpus_root)?;
    let train: Vec<u8> = b"the cat sat. ".iter().cycle().take(512).copied().collect();
    let heldout = b"a held-out story\x03tail";
    let entry = |bytes: &[u8]| json!({"bytes":bytes.len(),"sha256":digest(bytes)});
    let manifest = serde_json::to_vec(&json!({"schema":1,"kind":"byte-corpus",
        "files":{"train.txt":entry(&train),"heldout.txt":entry(heldout)}}))?;
    std::fs::write(corpus_root.join("train.txt"), &train)?;
    std::fs::write(corpus_root.join("heldout.txt"), heldout)?;
    std::fs::write(corpus_root.join("manifest.json"), &manifest)?;
    let mut corpus = Corpus::open(&corpus_root, &digest(&manifest), 32)?;
    corpus.window = Geometry::fixture().context;
    let options = Options {
        precision: Precision::Fp32,
        lanes: 2,
        workers: 2,
        resident_tiles: Some(RESIDENT),
        write_behind_tiles: 2,
        ..Options::default()
    };
    let mut config: Config = serde_json::from_value(json!({"schema":1,"out":root,
        "data":{"kind":"corpus","root":corpus_root,"manifest_sha256":digest(&manifest),
        "window":corpus.window},"seed":7,"rate":0.003,"anchors":8,"wall_seconds":60,
        "checkpoint_every":4,"eval_every":8,"eval_cases":1,"resume":false,"connected":options,"tiles":19}))?;
    // Paging keys are never serialized, so they reach the config only directly.
    config.connected = Some(options);
    Ok((corpus, config, options))
}

fn steps(state: &mut State, store: &mut Store, corpus: &Corpus, count: usize) -> Result<Vec<f32>> {
    let options = state.options;
    (0..count)
        .map(|_| connected_step::step(state, store, corpus, options, 0.003, false))
        .collect()
}

#[test]
fn tiles19_trains_paged_resumes_bit_exactly_and_completes() -> Result<()> {
    assert_eq!(TILES, 19);
    let root = io::Scratch::new()?;
    let (corpus, config, options) = corpus(&root.path)?;
    let _guard = tuning::scoped(config.knobs());
    let (geometry, identity) = (Geometry::fixture(), config.identity()?);
    let fresh = |name: &str| -> Result<(State, Store)> {
        let mut store = Store::new_precision(
            &root.path.join(name),
            geometry,
            7,
            options.workers,
            options.precision,
        )?;
        store.set_paging(RESIDENT, options.write_behind_tiles)?;
        Ok((State::new(geometry, 7, corpus.cursor(0)?, &options)?, store))
    };
    let (mut reference, mut reference_store) = fresh("reference")?;
    let expected = steps(&mut reference, &mut reference_store, &corpus, 8)?;
    // Four anchors, a checkpoint, then four more from the restored state and store.
    let (mut first, mut first_store) = fresh("connected-tiles")?;
    let mut losses = steps(&mut first, &mut first_store, &corpus, 4)?;
    checkpoint::save(
        &root.path,
        &mut first_store,
        &identity,
        4,
        &first.payloads()?,
    )?;
    drop((first, first_store));
    let loaded = checkpoint::resume(&root.path, geometry, &identity, options.workers)?.unwrap();
    let mut resumed = State::restore(&loaded.payloads, geometry, 7)?;
    resumed.options = resumed.options.with_paging(&options);
    let mut store = loaded.store;
    store.set_paging(RESIDENT, options.write_behind_tiles)?;
    losses.extend(steps(&mut resumed, &mut store, &corpus, 4)?);
    assert_eq!(losses, expected);
    assert_eq!(resumed.payloads()?, reference.payloads()?);
    store.flush()?;
    reference_store.flush()?;
    let meta = |store: &Store| {
        store
            .slots
            .iter()
            .map(|s| s.meta.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(meta(&store), meta(&reference_store));
    assert!(store.slots.len() == TILES && store.stats().peak_resident <= RESIDENT);
    // Routes left the resident set (so tiles were paged) and reached the outer ring (routes < 7).
    let counters = &resumed.counters;
    let visited: Vec<usize> = (0..TILES)
        .filter(|&tile| counters.anchor_visits[tile] + counters.local_visits[tile] > 0)
        .collect();
    assert!(visited.len() > RESIDENT, "only {visited:?} visited");
    let outer = |&tile: &usize| geometry.routes(tile).is_ok_and(|routes| routes < 7);
    assert!(
        visited.iter().any(outer),
        "no outer-ring tile in {visited:?}"
    );
    // Completion reads the 19-tile checkpoint and leaves it unchanged; its config names the
    // fixture window, which `Corpus::open` (floor 32) cannot stamp, so the cursor is relabelled.
    resumed.cursor.data.dataset_version = format!("window-{}", corpus.window);
    let name = checkpoint::save(&root.path, &mut store, &identity, 8, &resumed.payloads()?)?;
    let name = name.file_name().unwrap().to_str().unwrap().to_string();
    drop(store);
    let result = complete_geometry(&config, &name, "the ", 4, geometry)?;
    assert_eq!(result["before_sha256"], result["after_sha256"]);
    assert!(result["generated_bytes"].as_u64().unwrap() > 0);
    // A seven-tile manifest is refused by tile count, not misread, and left in place.
    let path = root
        .path
        .join("checkpoints")
        .join(&name)
        .join("manifest.json");
    let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    manifest["tiles"].as_array_mut().unwrap().truncate(7);
    std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
    let error = checkpoint::resume(&root.path, geometry, &identity, 2)
        .err()
        .unwrap();
    assert!(format!("{error:#}")
        .contains("has 7 tiles but this binary is built for 19; use the default build"));
    assert!(path.exists());
    Ok(())
}
