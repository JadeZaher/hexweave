//! Core checkpoint resume contracts: bit-exact fallback, quarantine policy, identity refusal.

use super::checkpoint_state::StateRecord;
use super::core_best;
use super::core_checkpoint;
use super::core_contracts::{assay, fingerprint, fresh_temp, train};
use super::core_drill::Drill;
use super::geometry::Geometry;
use super::resident::ResidentStore;
use super::state::State;
use super::tuning::{self, Knobs};
use anyhow::{Context, Result};
use serde_json::json;
use std::io::{Error as Io, ErrorKind};
use std::path::Path;

/// Resume is bit-exact; a corrupt newest checkpoint is set aside for the previous one, a stale
/// partial is ignored, any foreign identity is refused before anything is touched, only
/// integrity failures quarantine, and once all are quarantined resume refuses anchor 0.
#[test]
fn core_resume_is_bit_exact_and_falls_back_past_corruption() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    let geometry = Geometry::candidate();
    let (out, mixed) = (fresh_temp("resume")?, fresh_temp("mixed")?);
    let _core = tuning::scoped(Knobs::core(0.003, 7));
    let mut state = State::with_cursor(geometry, 7, drill.cursor(0)?)?;
    state.anchor_only = true;
    let mut store = ResidentStore::new(geometry, 7)?;
    train(&mut state, &mut store, &drill, 5)?;
    core_checkpoint::save(&out, &state, &store, "identity")?;
    let foreign = core_checkpoint::save(&mixed, &state, &store, "other")?;
    train(&mut state, &mut store, &drill, 5)?;
    let newest = core_checkpoint::save(&out, &state, &store, "identity")?;
    let mixed_newest = core_checkpoint::save(&mixed, &state, &store, "identity")?;
    train(&mut state, &mut store, &drill, 5)?;
    let uninterrupted = fingerprint(&state, &store)?;
    let corrupt = |checkpoint: &Path| -> Result<()> {
        let tiles = checkpoint.join("tiles.bin");
        let mut bytes = std::fs::read(&tiles)?;
        bytes[100] ^= 1;
        Ok(std::fs::write(&tiles, bytes)?)
    };
    let refused = |run: &Path| -> Result<bool> {
        let error = core_checkpoint::resume(run, geometry, "identity").err();
        Ok(format!("{:#}", error.context("resume succeeded")?).contains("different run identity"))
    };
    corrupt(&newest)?;
    corrupt(&mixed_newest)?;
    assert!(
        refused(&mixed)? && mixed_newest.exists(),
        "an older foreign checkpoint must stop resume before a corrupt newer one is quarantined"
    );
    let foreign_load = core_checkpoint::load(&foreign, geometry, "identity").err();
    for (error, quarantine) in [
        (anyhow::Error::from(Io::from(ErrorKind::NotFound)), true),
        (
            anyhow::Error::from(Io::from(ErrorKind::PermissionDenied)),
            false,
        ),
        (
            anyhow::Error::from(Io::from(ErrorKind::PermissionDenied)).context("tiles.bin"),
            false,
        ),
        (
            anyhow::anyhow!("checkpoint file tiles.bin hash mismatch"),
            true,
        ),
        (foreign_load.context("foreign load succeeded")?, false),
    ] {
        assert_eq!(
            core_checkpoint::integrity_failure(&error),
            quarantine,
            "{error:#}"
        );
    }
    std::fs::remove_dir_all(&mixed)?;
    let partial = out.join("checkpoints").join("a000000000015.partial");
    std::fs::create_dir_all(&partial)?;
    let error = core_checkpoint::resume(&out, geometry, "other").err();
    assert!(format!("{:#}", error.context("foreign resume succeeded")?)
        .contains("different run identity"));
    assert!(
        newest.exists(),
        "a foreign identity must not quarantine checkpoints"
    );
    tuning::install(Knobs::core(0.003, 7));
    let (mut resumed, mut store, path) =
        core_checkpoint::resume(&out, geometry, "identity")?.context("no checkpoint resumed")?;
    assert!(path.ends_with("a000000000005") && !newest.exists());
    assert!(
        partial.exists(),
        "resume must leave a stale partial in place"
    );
    train(&mut resumed, &mut store, &drill, 10)?;
    assert!(
        fingerprint(&resumed, &store)? == uninterrupted,
        "fallback resume must be bit-exact"
    );
    let mut tampered = serde_json::to_value(StateRecord::capture(&resumed)?)?;
    tampered["global_opportunity"] = json!(resumed.global_opportunity + 1);
    let record: StateRecord = serde_json::from_value(tampered)?;
    assert!(
        record.restore(resumed.shared, resumed.predictor).is_err(),
        "restore must reject a clock that disagrees with the cursor"
    );
    corrupt(&path)?;
    for attempt in [
        "quarantines the last checkpoint",
        "finds only quarantined ones",
    ] {
        assert!(
            core_checkpoint::resume(&out, geometry, "identity").is_err(),
            "resume that {attempt} must refuse a fresh start"
        );
    }
    std::fs::remove_dir_all(&out)?;
    Ok(())
}

/// Keep-best: a better score is saved under out/best with an identity-stamped selection; a better
/// repeat at an anchor that already has a best checkpoint (resume after a crash between save and
/// selection) still records; another run's selection is ignored; a fresh run refuses the directory.
#[test]
fn best_checkpoint_selection_survives_repeats_and_ignores_other_runs() -> Result<()> {
    let source = assay()?;
    let drill = Drill::new(&source, 7)?;
    let geometry = Geometry::candidate();
    let out = fresh_temp("best")?;
    let _core = tuning::scoped(Knobs::core(0.003, 7));
    let mut state = State::with_cursor(geometry, 7, drill.cursor(0)?)?;
    state.anchor_only = true;
    let mut store = ResidentStore::new(geometry, 7)?;
    train(&mut state, &mut store, &drill, 1)?;
    let mut best = core_best::best_so_far(&out, "identity");
    for score in [2.0, 1.0, 3.0] {
        let line = json!({ "eval": { "dev_nll_per_byte": score } });
        core_best::keep(&out, &state, &store, "identity", &line, &mut best);
    }
    assert_eq!(
        best, 1.0,
        "the better repeat at the same anchor must be kept"
    );
    assert_eq!(core_best::best_so_far(&out, "identity"), 1.0);
    assert_eq!(core_best::best_so_far(&out, "other"), f64::INFINITY);
    assert!(out
        .join("best/checkpoints/a000000000001/manifest.json")
        .exists());
    assert!(core_best::ensure_fresh(&out).is_err());
    std::fs::remove_dir_all(&out)?;
    Ok(())
}
