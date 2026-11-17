use std::process::{Command, Output};

fn invoke(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hexweave"))
        .args(args)
        .output()
        .expect("fixture binary should start")
}

#[test]
fn train_and_complete_reject_bad_invocations_before_work() {
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
        (vec!["complete"], "expected complete --config"),
        (
            vec!["complete", "--config"],
            "each complete option needs a value",
        ),
        (
            vec!["complete", "--prompt", "x", "--prompt", "y"],
            "unknown or duplicate",
        ),
        (vec!["complete", "--feedback", "sample"], "feedback must be"),
        (
            vec!["complete", "--trace-bytes", "many"],
            "trace-bytes must be",
        ),
        (
            vec![
                "complete",
                "--config",
                "missing.json",
                "--checkpoint",
                "a000000000000",
                "--prompt",
                "X",
                "--trace-bytes",
                "17",
            ],
            "trace-bytes must be",
        ),
        (
            vec![
                "complete",
                "--config",
                "missing.json",
                "--checkpoint",
                "a000000000000",
                "--prompt",
                "X",
                "--feedback",
                "prompt-only",
            ],
            "requires a nonzero",
        ),
        (
            vec![
                "complete",
                "--config",
                "missing.json",
                "--checkpoint",
                "a000000000000",
                "--prompt",
                "X",
                "--seed",
                "3",
            ],
            "apply only with --top-p",
        ),
        (
            vec![
                "complete",
                "--config",
                "missing.json",
                "--checkpoint",
                "a000000000000",
                "--prompt",
                "X",
                "--top-p",
                "1.5",
            ],
            "top-p must be in (0, 1]",
        ),
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

/// The child runs in a temporary directory, so its relative `out` never touches this checkout.
#[test]
fn invalid_kernel_fails_train_before_any_checkpoint() {
    let root = std::env::temp_dir().join(format!("hexweave-cli-kernel-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("temporary root");
    let out = format!("work/core-runs/cli-kernel-{}", std::process::id());
    let data =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/retained-text-assay");
    let config = root.join("run.json");
    let value = serde_json::json!({"schema": 1, "out": out, "data": {"kind": "drill", "root": data},
        "seed": 7, "rate": 0.003, "anchors": 1, "wall_seconds": 60, "checkpoint_every": 1,
        "eval_every": 1, "eval_cases": 1, "resume": false});
    std::fs::write(&config, value.to_string()).expect("temporary config");
    let config = config.to_string_lossy().into_owned();
    let output = Command::new(env!("CARGO_BIN_EXE_hexweave"))
        .args(["train", "--config", config.as_str()])
        .current_dir(&root)
        .env("HEXWEAVE_KERNEL", "invalid")
        .output()
        .expect("fixture binary should start");
    let checkpointed = root.join(&out).join("checkpoints").exists();
    std::fs::remove_dir_all(&root).expect("remove temporary root");
    assert!(!output.status.success() && !checkpointed);
    assert!(String::from_utf8_lossy(&output.stderr).contains("HEXWEAVE_KERNEL"));
}
