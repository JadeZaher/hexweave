use anyhow::Result;
use hexweave::text::{run, CreditDiagnostics, Mode, Options, Profile, RouteCredit};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn directory(label: &str) -> Result<PathBuf> {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    Ok(std::env::current_dir()?
        .join("work")
        .join(format!("text-{label}-{}-{unique}", std::process::id())))
}

#[test]
fn rejects_unbounded_steps_and_shared_training_validation() -> Result<()> {
    let invalid = Options {
        steps: 1025,
        ..Options::default()
    };
    assert!(invalid.validate().is_err());
    let file = Path::new("datasets/tiny/train.txt");
    assert!(run(&Options::default(), file, file, &directory("invalid")?).is_err());
    Ok(())
}

#[test]
fn disk_backed_text_is_bounded_and_reproducible() -> Result<()> {
    let options = Options {
        steps: 8,
        ..Options::default()
    };
    let train = Path::new("datasets/tiny/train.txt");
    let valid = Path::new("datasets/tiny/valid.txt");
    let first_dir = directory("first")?;
    let first = run(&options, train, valid, &first_dir)?;
    let second = run(&options, train, valid, &directory("second")?)?;
    assert_eq!(first["status"], "complete");
    assert_eq!(first["protocol"], "text-pulse-v4");
    assert_eq!(
        first["architecture"]["training_pulse"]["kind"],
        "joint-tile-pulse-v1"
    );
    assert_eq!(first["counts"], second["counts"]);
    assert_eq!(first["evaluation"], second["evaluation"]);
    assert_eq!(first["counts"]["anchor_targets"], 2);
    assert_eq!(first["residency"]["peak_live_tiles"], 1);
    assert_eq!(
        first["residency"]["inactive_tiles_application_loaded"],
        false
    );
    assert!(first["counts"]["local_updates"].as_u64().unwrap() > 0);
    assert!(first["tile_transitions"]["pulse_context"].as_u64().unwrap() > 0);
    assert!(first["tile_transitions"]["pulse_heads"].as_u64().unwrap() > 0);
    assert!(first["disk_io"]["bytes_read"].as_u64().unwrap() > 0);
    assert!(first["disk_io"]["bytes_written"].as_u64().unwrap() > 0);
    assert!(first_dir.join("snapshot.json").is_file());
    let detached_dir = directory("detached")?;
    let detached_options = Options {
        route_credit: RouteCredit::RouteToUDetached,
        ..options.clone()
    };
    let detached = run(&detached_options, train, valid, &detached_dir)?;
    assert_eq!(
        first["architecture"]["parameter_count"],
        detached["architecture"]["parameter_count"]
    );
    assert_eq!(
        first["evaluation"]["initial_mean_nll_nats_per_byte"],
        detached["evaluation"]["initial_mean_nll_nats_per_byte"]
    );
    assert_eq!(detached["options"]["route_credit"], "route-to-u-detached");
    let snapshot: serde_json::Value =
        serde_json::from_slice(&std::fs::read(detached_dir.join("snapshot.json"))?)?;
    assert_eq!(snapshot["options"], detached["options"]);
    assert_eq!(snapshot["protocol"], detached["protocol"]);
    assert_eq!(snapshot["resumable"], false);
    for id in 0..7 {
        let tile = format!("tile-{id}-v0.json");
        assert_eq!(
            std::fs::read(first_dir.join(&tile))?,
            std::fs::read(detached_dir.join(&tile))?
        );
    }
    assert!(run(&options, train, valid, &first_dir).is_err());
    Ok(())
}

#[test]
fn periodic_anchor_control_never_updates_predictor() -> Result<()> {
    let options = Options {
        steps: 5,
        mode: Mode::AnchorOnly,
        ..Options::default()
    };
    let result = run(
        &options,
        Path::new("datasets/tiny/train.txt"),
        Path::new("datasets/tiny/valid.txt"),
        &directory("control")?,
    )?;
    assert_eq!(result["counts"]["anchor_targets"], 2);
    assert_eq!(result["counts"]["local_updates"], 0);
    assert_eq!(result["counts"]["predictor_updates"], 0);
    assert_eq!(result["shared_transitions"]["predictor"], 0);
    Ok(())
}

#[test]
fn byte_validation_target_does_not_change_training_weights() -> Result<()> {
    let options = Options {
        steps: 4,
        ..Options::default()
    };
    let train = Path::new("datasets/tiny/train.txt");
    let first_dir = directory("valid-a")?;
    let second_dir = directory("valid-b")?;
    let first = run(
        &options,
        train,
        Path::new("datasets/tiny/valid.txt"),
        &first_dir,
    )?;
    let second = run(
        &options,
        train,
        Path::new("datasets/tiny/README.md"),
        &second_dir,
    )?;
    assert_eq!(first["counts"], second["counts"]);
    assert_eq!(
        std::fs::read(first_dir.join("shared-final.json"))?,
        std::fs::read(second_dir.join("shared-final.json"))?
    );
    Ok(())
}

#[test]
fn t2_diagnostic_toggle_preserves_routes_and_exposure_and_rejects_old_validation() -> Result<()> {
    for arm in [RouteCredit::Coupled, RouteCredit::RouteToUDetached] {
        let mut options = Options {
            steps: 8,
            profile: Profile::T2Credit,
            route_credit: arm,
            ..Options::default()
        };
        let train = Path::new("datasets/tiny/train.txt");
        let valid = Path::new("datasets/tiny/development-t2.txt");
        assert!(run(
            &options,
            train,
            Path::new("datasets/tiny/valid.txt"),
            &directory("t2-wrong")?
        )
        .is_err());
        let off_dir = directory("t2-off")?;
        let on_dir = directory("t2-on")?;
        let off = run(&options, train, valid, &off_dir)?;
        options.credit_diagnostics = CreditDiagnostics::On;
        let on = run(&options, train, valid, &on_dir)?;
        for field in [
            "counts",
            "evaluation",
            "tile_transitions",
            "shared_transitions",
            "disk_io",
        ] {
            assert_eq!(off[field], on[field], "diagnostic changed {field}");
        }
        assert_eq!(
            std::fs::read(off_dir.join("shared-final.json"))?,
            std::fs::read(on_dir.join("shared-final.json"))?
        );
        assert_eq!(on["data"]["evaluation_prefix"]["byte_count"], 72);
        assert_eq!(on["credit_diagnostics"]["anchors"], 2);
        assert_eq!(
            on["credit_diagnostics"]["total"]["raw"]["frames"],
            on["counts"]["returned_frames"]
        );
        assert!(off["credit_diagnostics"].is_null());
        for (receipt, shadows) in [(&off, 0), (&on, 3)] {
            let residency = &receipt["residency"];
            assert_eq!(
                residency["maximum_actual_update_gradient_bundle_bytes"],
                6656
            );
            assert_eq!(residency["diagnostic_shadow_gradient_bundles"], shadows);
            assert_eq!(
                residency["maximum_diagnostic_shadow_gradient_bytes"],
                shadows * 6656
            );
            assert_eq!(
                residency["maximum_concurrent_gradient_bundle_bytes"],
                (1 + shadows) * 6656
            );
            assert_eq!(residency["physical_rss_measured_here"], false);
        }
    }
    Ok(())
}
