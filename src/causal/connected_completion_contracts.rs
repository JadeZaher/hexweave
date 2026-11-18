//! Connected completion freeze, rejection and scratch-cleanup contracts; see CONNECTED.md.

use super::super::{connected_config, connected_step};
use super::*;

struct Fixture {
    config: Config,
    root: io::Scratch,
    name: String,
}

fn fixture() -> Result<Fixture> {
    fixture_precision(super::super::connected_precision::Precision::Ternary)
}

fn fixture_precision(precision: super::super::connected_precision::Precision) -> Result<Fixture> {
    fixture_with(connected_config::Options {
        precision,
        lanes: 2,
        workers: 2,
        visits: 2,
        ..connected_config::Options::default()
    })
}

/// One trained anchor checkpointed under `options`; the config carries its paging controls too.
fn fixture_with(options: connected_config::Options) -> Result<Fixture> {
    let root = io::Scratch::new()?;
    let corpus = root.path.join("corpus");
    std::fs::create_dir(&corpus)?;
    let heldout = b"a held-out story\x03tail";
    let entry = |bytes: &[u8]| json!({"bytes":bytes.len(),"sha256":digest(bytes)});
    let manifest = serde_json::to_vec(&json!({"schema":1,"kind":"byte-corpus",
        "files":{"train.txt":entry(b"train"),"heldout.txt":entry(heldout)}}))?;
    std::fs::write(corpus.join("train.txt"), b"train")?;
    std::fs::write(corpus.join("heldout.txt"), heldout)?;
    std::fs::write(corpus.join("manifest.json"), &manifest)?;
    let manifest = digest(&manifest);
    let config: Config = serde_json::from_value(json!({"schema":1,"out":root.path,
        "data":{"kind":"corpus","root":corpus,
        "manifest_sha256":manifest,"window":8},"seed":7,"rate":0.003,
        "anchors":8,"wall_seconds":60,"checkpoint_every":8,"eval_every":8,
        "eval_cases":1,"resume":false,"connected":options}))?;
    let mut config = config;
    // Paging keys are never serialized, so they reach the config only directly.
    config.connected = Some(options);
    tuning::install(config.knobs());
    let source = fixture::Source::new();
    let mut state = State::new(Geometry::fixture(), 7, source.cursor(), &options)?;
    let mut store = Store::new_precision(
        &root.path.join("training-runtime"),
        Geometry::fixture(),
        7,
        2,
        options.precision,
    )?;
    connected_step::step(&mut state, &mut store, &source, options, 0.003, false)?;
    state.cursor.data.schema = "byte-corpus-v1".into();
    state.cursor.data.manifest_sha256 = manifest;
    state.cursor.data.dataset_version = "window-8".into();
    let checkpoint = connected_checkpoint::save(
        &root.path,
        &mut store,
        &config.identity()?,
        state.anchors,
        &state.payloads()?,
    )?;
    let name = checkpoint
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    std::fs::write(
        root.path.join("STOP"),
        b"existing stopped training receipt\n",
    )?;
    Ok(Fixture { config, root, name })
}

#[test]
fn connected_completion_preserves_checkpoint_optimizer_and_existing_stop() -> Result<()> {
    let _guard = tuning::scoped(tuning::Knobs::core(0.003, 7));
    let fixture = fixture()?;
    let checkpoint = fixture.root.path.join("checkpoints").join(&fixture.name);
    let source_before = io::checkpoint_hash(&checkpoint, Geometry::fixture())?;
    let stop_before = std::fs::read(fixture.root.path.join("STOP"))?;
    std::fs::write(fixture.root.path.join("OWNER.lock"), b"active fixture")?;
    let result = complete_geometry(&fixture.config, &fixture.name, "X", 3, Geometry::fixture())?;
    assert_eq!(result["owner_lock_present"], true);
    assert_eq!(result["anchors"], 1);
    assert_eq!(result["before_sha256"], result["after_sha256"]);
    assert_eq!(
        result["checkpoint_before_sha256"],
        result["checkpoint_after_sha256"]
    );
    assert_eq!(result["checkpoint_before_sha256"], source_before);
    assert_eq!(result["scratch_removed"], true);
    assert_eq!(result["corpus_opened"], true);
    assert_eq!(result["context_bytes"], 7, "the prompt ends a full window");
    assert_ne!(result["stop_reason"], "inference_deadline");
    assert!(result["generated_bytes"].as_u64().unwrap() <= 3);
    assert!(result["paging"]["peak_resident"].as_u64().unwrap() <= 2);
    assert_eq!(
        io::checkpoint_hash(&checkpoint, Geometry::fixture())?,
        source_before
    );
    assert_eq!(std::fs::read(fixture.root.path.join("STOP"))?, stop_before);
    let scratch = io::Scratch::new()?;
    let snapshot = scratch.path.join("checkpoints").join(&fixture.name);
    assert_eq!(
        io::snapshot(&checkpoint, &snapshot, Geometry::fixture())?,
        source_before
    );
    std::fs::remove_dir_all(&checkpoint)?;
    let expired = infer_snapshot(
        &fixture.config,
        &snapshot,
        &source_before,
        "X",
        3,
        Geometry::fixture(),
        &scratch.path,
        Instant::now().checked_sub(TIMEOUT).unwrap(),
        super::super::connected_decode::Options::default(),
    )?;
    assert_eq!(expired["stop_reason"], "inference_deadline");
    assert_eq!(expired["generated_bytes"], 0);
    assert_eq!(expired["before_sha256"], expired["after_sha256"]);
    Ok(())
}

#[test]
fn connected_completion_rejects_corruption_and_identity_without_source_writes() -> Result<()> {
    let _guard = tuning::scoped(tuning::Knobs::core(0.003, 7));
    let mut fixture = fixture()?;
    let checkpoint = fixture.root.path.join("checkpoints").join(&fixture.name);
    let before = io::checkpoint_hash(&checkpoint, Geometry::fixture())?;
    fixture.config.seed = 8;
    assert!(
        complete_geometry(&fixture.config, &fixture.name, "X", 1, Geometry::fixture()).is_err()
    );
    fixture.config.seed = 7;
    assert!(complete_geometry_options(
        &fixture.config,
        &fixture.name,
        "X",
        1,
        Geometry::fixture(),
        super::super::connected_decode::Options {
            quantize_audit: true,
            ..Default::default()
        }
    )
    .is_err());
    assert_eq!(
        io::checkpoint_hash(&checkpoint, Geometry::fixture())?,
        before
    );
    let tile = checkpoint.join("tiles/6.bin");
    let mut corrupt = std::fs::read(&tile)?;
    *corrupt.last_mut().unwrap() ^= 1;
    std::fs::write(&tile, &corrupt)?;
    let corrupt_hash = io::checkpoint_hash(&checkpoint, Geometry::fixture())?;
    let scratch = io::Scratch::new()?;
    let scratch_path = scratch.path.clone();
    assert!(infer(
        &fixture.config,
        &fixture.name,
        "X",
        1,
        Geometry::fixture(),
        &scratch.path,
        Instant::now(),
        super::super::connected_decode::Options::default()
    )
    .is_err());
    drop(scratch);
    assert!(!scratch_path.exists());
    assert_eq!(
        io::checkpoint_hash(&checkpoint, Geometry::fixture())?,
        corrupt_hash
    );
    assert_eq!(std::fs::read(&tile)?, corrupt);
    assert!(checkpoint.is_dir());
    Ok(())
}

#[test]
fn fp32_quantization_comparison_preserves_masters_and_resumes_the_named_precision() -> Result<()> {
    let _guard = tuning::scoped(tuning::Knobs::core(0.003, 7));
    let mut fixture = fixture_precision(super::super::connected_precision::Precision::Fp32)?;
    let checkpoint = fixture.root.path.join("checkpoints").join(&fixture.name);
    let before = io::checkpoint_hash(&checkpoint, Geometry::fixture())?;
    let stop = std::fs::read(fixture.root.path.join("STOP"))?;
    let result = complete_geometry_options(
        &fixture.config,
        &fixture.name,
        "X",
        2,
        Geometry::fixture(),
        super::super::connected_decode::Options {
            trace_bytes: 2,
            quantize_audit: true,
            ..Default::default()
        },
    )?;
    assert_eq!(result["protocol"], "causal-connected-fp32-v5");
    assert_eq!(result["before_sha256"], result["after_sha256"]);
    assert_eq!(result["checkpoint_before_sha256"], before);
    assert_eq!(result["checkpoint_after_sha256"], before);
    assert_eq!(result["scratch_removed"], true);
    assert_eq!(result["master_weights_preserved"], true);
    let variants = result["variants"].as_array().unwrap();
    assert_eq!(variants.len(), 3);
    for (i, name) in ["fp32", "int8", "ternary"].into_iter().enumerate() {
        assert_eq!(variants[i]["weight_view"], name);
        assert_ne!(variants[i]["stop_reason"], "inference_deadline");
        let traces = variants[i]["decode_trace"].as_array().unwrap();
        assert!(!traces.is_empty());
        assert!(traces
            .iter()
            .all(|trace| trace["scalar_readout_max_abs_error"].as_f64().unwrap() < 1e-6));
        let matrices = variants[i]["quantization"]["matrices"].as_array().unwrap();
        // Seven tiles of eleven matrices plus the tied embedding.
        #[cfg(not(feature = "tiles19"))]
        assert_eq!(matrices.len(), 78);
        if i == 0 {
            assert!(matrices.iter().all(|matrix| matrix["mse"] == 0.0));
        } else {
            assert!(matrices
                .iter()
                .any(|matrix| matrix["mse"].as_f64().unwrap() > 0.0));
        }
    }
    assert!(result["paging"]["peak_resident"].as_u64().unwrap() <= 2);
    let largest = Geometry::fixture().tile_coefficients(super::super::geometry::START_TILE)?;
    assert!(result["paging"]["peak_view_bytes"].as_u64().unwrap() <= (2 * largest * 4) as u64);
    assert_eq!(std::fs::read(fixture.root.path.join("STOP"))?, stop);
    reject_partial_masters(&fixture, &checkpoint)?;
    fixture.config.connected.as_mut().unwrap().precision =
        super::super::connected_precision::Precision::Ternary;
    assert!(
        complete_geometry(&fixture.config, &fixture.name, "X", 1, Geometry::fixture()).is_err()
    );
    assert_eq!(
        io::checkpoint_hash(&checkpoint, Geometry::fixture())?,
        before
    );
    Ok(())
}

#[test]
fn connected_completion_accepts_a_checkpoint_trained_with_paging_controls() -> Result<()> {
    let _guard = tuning::scoped(tuning::Knobs::core(0.003, 7));
    // The checkpoint omits paging keys; completion must compare options without them.
    let fixture = fixture_with(connected_config::Options {
        lanes: 2,
        workers: 2,
        visits: 2,
        resident_tiles: Some(3),
        write_behind_tiles: 2,
        ..connected_config::Options::default()
    })?;
    let result = complete_geometry(&fixture.config, &fixture.name, "X", 3, Geometry::fixture())?;
    assert_eq!(result["before_sha256"], result["after_sha256"]);
    Ok(())
}

fn reject_partial_masters(fixture: &Fixture, checkpoint: &Path) -> Result<()> {
    let mut fresh = State::new(
        Geometry::fixture(),
        7,
        fixture::Source::new().cursor(),
        &fixture.config.connected.unwrap(),
    )?;
    fresh.shared_latent = None;
    assert!(fresh.payloads().is_err());
    for retained in [0, 11] {
        let scratch = io::Scratch::new()?;
        let mut loaded = connected_checkpoint::load(
            checkpoint,
            &scratch.path,
            Geometry::fixture(),
            &fixture.config.identity()?,
            2,
        )?;
        let mut lease = loaded.store.checkout(3)?;
        lease.parts_mut()?.1.truncate(retained);
        assert!(loaded.store.commit(lease).is_err());
        assert!(loaded.store.quiescent().is_err());
    }
    Ok(())
}
