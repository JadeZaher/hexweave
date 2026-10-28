use std::process::{Command, Output};

fn invoke(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hexweave"))
        .args(args)
        .output()
        .expect("fixture binary should start")
}

#[test]
fn train_rejects_bad_invocations_and_configs_before_training() {
    let config = std::env::temp_dir().join(format!("hexweave-cli-{}.json", std::process::id()));
    std::fs::write(
        &config,
        r#"{"schema":1,"out":"work/core-runs/../../elsewhere","data":{"kind":"drill","root":"datasets/retained-text-assay"},
            "seed":7,"rate":0.003,"anchors":1,"wall_seconds":60,"checkpoint_every":1,"eval_every":1,
            "eval_cases":1,"resume":false}"#,
    )
    .expect("temporary config");
    let path = config.to_string_lossy().into_owned();
    for (args, expected) in [
        (vec!["train"], "expected train --config"),
        (vec!["train", "--config"], "expected train --config"),
        (
            vec!["train", "--config", path.as_str()],
            "out must be work/core-runs/<name>",
        ),
        (vec!["resume"], "unknown command"),
        (vec!["surprise"], "unknown command"),
    ] {
        let output = invoke(&args);
        assert!(!output.status.success(), "accepted {args:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
    }
    assert!(!std::path::Path::new("elsewhere").exists());
    std::fs::remove_file(config).expect("remove temporary config");
}

#[test]
fn smoke_rejects_invalid_duplicate_and_unbounded_options() {
    for args in [
        vec!["smoke", "--steps", "0"],
        vec!["smoke", "--steps", "17"],
        vec!["smoke", "--steps", "184467440737095516160"],
        vec!["smoke", "--steps", "1", "--steps", "1"],
        vec!["smoke", "--seed", "7", "--seed", "7"],
        vec!["smoke", "--seed"],
        vec!["smoke", "--config", "configs/local-287m.json"],
    ] {
        let output = invoke(&args);
        assert!(!output.status.success(), "accepted {args:?}");
        assert!(
            output.stdout.is_empty(),
            "unexpected success output for {args:?}"
        );
    }
}

#[test]
fn plan_requires_exact_profile_option() {
    for args in [
        vec!["plan"],
        vec!["plan", "configs/tiny.json"],
        vec!["plan", "--config", "configs/tiny.json", "--extra"],
    ] {
        let output = invoke(&args);
        assert!(!output.status.success());
    }
}

#[test]
fn help_returns_success_without_training() {
    let output = invoke(&["--help"]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("hexweave train --config"));
}

#[test]
fn pulse_rejects_unbounded_unsupported_and_duplicate_options() {
    for args in [
        vec!["pulse", "--steps", "257"],
        vec!["pulse", "--steps", "0"],
        vec!["pulse", "--residents", "4"],
        vec!["pulse", "--anchor-interval", "0"],
        vec!["pulse", "--mode", "production"],
        vec!["pulse", "--seed", "7", "--seed", "8"],
        vec!["pulse", "--config", "configs/local-287m.json"],
        vec!["pulse", "--mode"],
    ] {
        let output = invoke(&args);
        assert!(!output.status.success(), "accepted {args:?}");
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn pulse_command_returns_its_exact_bounded_identity() {
    let output = invoke(&[
        "pulse",
        "--steps",
        "1",
        "--seed",
        "17",
        "--mode",
        "anchor-only",
        "--residents",
        "2",
        "--anchor-interval",
        "3",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["protocol"], "ternary-pulse-v1");
    assert_eq!(result["completed_steps"], 1);
    assert_eq!(result["seed"], 17);
    assert_eq!(result["mode"], "anchor-only");
    assert_eq!(result["resident_limit"], 2);
    assert_eq!(result["anchor_interval"], 3);
}

#[test]
fn kernel_selection_is_process_local_and_numerically_equivalent() {
    let run = |kernel| {
        Command::new(env!("CARGO_BIN_EXE_hexweave"))
            .args(["pulse", "--steps", "16"])
            .env("HEXWEAVE_KERNEL", kernel)
            .output()
            .unwrap()
    };
    let invalid = run("invalid");
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("HEXWEAVE_KERNEL"));
    let scalar = run("scalar");
    assert!(scalar.status.success());
    let mut reference: serde_json::Value = serde_json::from_slice(&scalar.stdout).unwrap();
    reference.as_object_mut().unwrap().remove("seconds");
    reference["arithmetic"]
        .as_object_mut()
        .unwrap()
        .remove("execution");
    for kernel in ["auto", "avx2"] {
        let output = run(kernel);
        if kernel == "avx2" && !hexweave::ternary::avx2_available() {
            assert!(!output.status.success());
            continue;
        }
        assert!(output.status.success());
        let mut actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        actual.as_object_mut().unwrap().remove("seconds");
        actual["arithmetic"]
            .as_object_mut()
            .unwrap()
            .remove("execution");
        assert_eq!(actual, reference);
    }
}

#[test]
fn text_rejects_unbounded_missing_and_duplicate_options() {
    for args in [
        vec!["text", "--steps", "1025"],
        vec!["text", "--steps", "0"],
        vec!["text", "--mode", "full"],
        vec!["text", "--profile", "t1"],
        vec!["text", "--credit-diagnostics", "on"],
        vec![
            "text",
            "--profile",
            "t2-credit",
            "--credit-diagnostics",
            "unknown",
        ],
        vec!["text", "--route-credit", "uncoupled"],
        vec![
            "text",
            "--route-credit",
            "coupled",
            "--route-credit",
            "route-to-u-detached",
        ],
        vec!["text", "--seed", "1", "--seed", "2"],
        vec!["text", "--train"],
        vec!["text", "--config", "configs/local-287m.json"],
        vec!["text"],
    ] {
        let output = invoke(&args);
        assert!(!output.status.success(), "accepted {args:?}");
        assert!(output.stdout.is_empty());
    }
}
