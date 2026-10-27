use anyhow::Result;
use hexweave::pulse::{run, run_with_options, Mode, Options};
use serde_json::Value;

fn deterministic(mut report: Value) -> Value {
    report.as_object_mut().unwrap().remove("seconds");
    report
}

#[test]
fn pulse_is_repeatable_and_every_parameter_family_really_changes() -> Result<()> {
    let report = run(48, 7, Mode::Pulse)?;
    assert_eq!(
        deterministic(report.clone()),
        deterministic(run(48, 7, Mode::Pulse)?)
    );
    assert_eq!(report["protocol"], "ternary-pulse-v1");
    assert_eq!(report["all_learned_weights_ternary"], true);
    assert_eq!(report["fp32_master_weights"], false);
    for family in [
        "embedding",
        "readout",
        "transform",
        "activation",
        "router",
        "predictor",
    ] {
        assert!(
            report["ternary_transitions"][family].as_u64().unwrap() > 0,
            "{family} never changed"
        );
    }
    assert!(report["initial_loss"].as_f64().unwrap().is_finite());
    assert!(report["final_loss"].as_f64().unwrap().is_finite());
    Ok(())
}

#[test]
fn target_exposure_and_sampling_do_not_alias_anchor_cadence() -> Result<()> {
    let report = run(48, 7, Mode::Pulse)?;
    assert_eq!(report["training_target_exposures"], 12);
    assert_eq!(report["evaluation_target_exposures"], 8);
    for count in report["counts"]["anchor_targets_per_training_input"]
        .as_array()
        .unwrap()
    {
        assert_eq!(count, 1);
    }
    assert_eq!(report["counts"]["predicted_steps"], 36);
    assert_eq!(report["completed_steps"], 48);
    Ok(())
}

#[test]
fn anchor_ablation_skips_local_work_and_shares_initialization_and_task_exposure() -> Result<()> {
    let pulse = run(16, 7, Mode::Pulse)?;
    let anchor = run(16, 7, Mode::AnchorOnly)?;
    assert_eq!(pulse["initial_loss"], anchor["initial_loss"]);
    assert_eq!(
        pulse["training_target_exposures"],
        anchor["training_target_exposures"]
    );
    assert_eq!(anchor["counts"]["skipped_local_opportunities"], 12);
    assert_eq!(anchor["counts"]["local_node_updates"], 0);
    assert_eq!(anchor["counts"]["predictor_forward_calls"], 0);
    assert_eq!(anchor["history"]["peak_records"], 0);
    assert_eq!(anchor["ternary_transitions"]["predictor"], 0);
    Ok(())
}

#[test]
fn all_resident_limits_and_history_bounds_hold() -> Result<()> {
    for resident_limit in 1..=3 {
        let report = run_with_options(Options {
            steps: 48,
            resident_limit,
            ..Options::default()
        })?;
        assert!(report["residency"]["peak_nodes"].as_u64().unwrap() <= resident_limit as u64);
        assert!(report["history"]["peak_records"].as_u64().unwrap() <= 8);
        let bytes = &report["state_bytes"];
        assert!(
            bytes["resident_integer_evidence_peak"].as_u64().unwrap()
                <= bytes["resident_integer_evidence_bound"].as_u64().unwrap()
        );
        assert!(
            bytes["history_record_payload_peak"].as_u64().unwrap()
                <= bytes["history_record_payload_bound"].as_u64().unwrap()
        );
    }
    Ok(())
}

#[test]
fn invalid_public_bounds_fail_before_running() {
    for steps in [0, 257, usize::MAX] {
        assert!(run(steps, 7, Mode::Pulse).is_err());
    }
    for resident_limit in [0, 4, usize::MAX] {
        assert!(run_with_options(Options {
            resident_limit,
            ..Options::default()
        })
        .is_err());
    }
    for anchor_interval in [0, 17, usize::MAX] {
        assert!(run_with_options(Options {
            anchor_interval,
            ..Options::default()
        })
        .is_err());
    }
}

#[test]
fn adversarial_seeds_and_maximum_interval_remain_bounded() -> Result<()> {
    for seed in [0, u64::MAX, 0x9e37_79b9_7f4a_7c15] {
        let report = run_with_options(Options {
            steps: 17,
            seed,
            anchor_interval: 16,
            ..Options::default()
        })?;
        assert_eq!(report["training_target_exposures"], 2);
        assert!(report["final_loss"].as_f64().unwrap().is_finite());
    }
    Ok(())
}
