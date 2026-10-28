"""Run the explicitly selected T2 diagnostic; T1's campaign budget is consumed."""

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

from scripts import text_runner
from scripts.pulse_io import digest, event, finite_tree, publish
from scripts.text_io import owned_parent
from scripts.text_resources import CHILD_LIMIT
from scripts.text_credit_io import check_credit
from scripts.text_profile import T2_DEVELOPMENT_SHA256, T2_PREFIX_SHA256, T2_TRAIN_SHA256, check_profile_receipt


SCHEDULE = ((7, "coupled"), (7, "route-to-u-detached"),
            (17, "route-to-u-detached"), (17, "coupled"),
            (29, "coupled"), (29, "route-to-u-detached"))
CAMPAIGN_SECONDS = 240
CHILD_SECONDS = 30
QUALITY_MARGIN = 0.01


def source_identity(root: Path) -> dict:
    paths = [root / "Cargo.toml", root / "Cargo.lock"]
    paths += sorted((root / "src").rglob("*.rs"))
    paths += sorted((root / "scripts").glob("*.py"))
    files = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or path.stat().st_size > 1_048_576:
            raise ValueError("source identity escapes workspace or exceeds bounded file size")
        files[path.relative_to(root).as_posix()] = digest(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files,
            "scope": "Rust source, Python owner/campaign source, Cargo manifest and lock",
            "binary_source_binding": "requires the separately recorded release build; hashes alone do not prove it"}


def measurement(directory: Path, owner: dict, seed: int, arm: str) -> dict:
    child = owner.get("child", {})
    if (owner.get("status") != "completed" or child.get("returncode") != 0 or
            child.get("reaped") is not True or child.get("terminal_peak_memory_queried") is not True or
            child.get("cleanup_errors") or not finite_tree(owner)):
        raise ValueError("campaign requires successful terminal cleanup and finite evidence")
    peak = child.get("memory", {}).get("peak_working_set_bytes")
    wall = child.get("wall_seconds")
    if type(peak) is not int or not 0 < peak <= CHILD_LIMIT or type(wall) not in (int, float) or wall <= 0:
        raise ValueError("campaign requires measured peak working set and child wall time")
    hashes = owner.get("input_sha256", {})
    if (set(hashes) != {"binary", "train", "valid"} or hashes != owner.get("input_sha256_after") or
            any(type(value) is not str or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value) for value in hashes.values())):
        raise ValueError("campaign requires stable binary/data SHA256 identities")
    result = owner["result"]
    expected = {"seed": seed, "route_credit": arm, "steps": 256, "anchor_interval": 4,
                "mode": "pulse", "profile": "t2-credit", "credit_diagnostics": "on"}
    if result["options"] != expected or owner.get("options") != expected:
        raise ValueError("campaign arm or exposure identity mismatch")
    if (result.get("protocol") != "text-pulse-v4" or hashes["train"] != T2_TRAIN_SHA256 or
            hashes["valid"] != T2_DEVELOPMENT_SHA256 or owner.get("evaluation_input", {}).get("prefix_sha256") != T2_PREFIX_SHA256):
        raise ValueError("T2 protocol or frozen development identity mismatch")
    check_profile_receipt(result, expected)
    check_credit(result, expected)
    evaluation = result["evaluation"]
    initial = evaluation["initial_mean_nll_nats_per_byte"]
    final = evaluation["final_mean_nll_nats_per_byte"]
    return {"seed": seed, "arm": arm, "receipt": str(directory / "report.json"),
            "initial_nll": initial, "final_nll": final, "improvement": initial - final,
            "child_wall_seconds": wall, "training_seconds": result["timings"]["training_seconds"],
            "peak_working_set_bytes": peak, "counts": result["counts"],
            "tile_transitions": result["tile_transitions"], "shared_transitions": result["shared_transitions"],
            "predictor_error": result["predictor_error"], "disk_io": result["disk_io"],
            "input_sha256": hashes, "resolved_kernel": result["resolved_kernel"],
            "evaluation_examples": evaluation["examples_per_pass"], "data": result["data"],
            "credit_diagnostics": result["credit_diagnostics"], "credit_integrity_checked": True,
            "terminal_cleanup": {key: child[key] for key in ("reaped", "returncode", "terminal_peak_memory_queried")}}


def paired_decision(runs: list[dict]) -> dict:
    if [(run["seed"], run["arm"]) for run in runs] != list(SCHEDULE):
        raise ValueError("all six receipts are required for a decision")
    if any(run.get("credit_integrity_checked") is not True for run in runs):
        raise ValueError("T2 decision requires verified credit metrics")
    pairs = []
    for seed in (7, 17, 29):
        arms = {run["arm"]: run for run in runs if run["seed"] == seed}
        a, b = arms["coupled"], arms["route-to-u-detached"]
        if a["initial_nll"] != b["initial_nll"]:
            raise ValueError("paired initial NLL mismatch")
        credit = a["credit_diagnostics"]
        pairs.append({"seed": seed, "a_improvement": a["improvement"], "b_improvement": b["improvement"],
                      "b_quality_advantage": b["improvement"] - a["improvement"],
                      "a_anchor_differences": credit["anchors_with_different_results"],
                      "a_task_descent": credit["total"]["raw"]["task_descent"],
                      "a_deficit_ratio": credit["total"]["ratios"]["deficit_over_task_descent"],
                      "a_to_b_wall_ratio": a["child_wall_seconds"] / b["child_wall_seconds"],
                      "a_to_b_training_ratio": a["training_seconds"] / b["training_seconds"]})
    go = all(pair["b_improvement"] >= QUALITY_MARGIN and pair["b_quality_advantage"] >= QUALITY_MARGIN and
             pair["a_anchor_differences"] >= 16 and pair["a_task_descent"] > 0 and
             pair["a_deficit_ratio"] is not None and pair["a_deficit_ratio"] >= 0.10 for pair in pairs)
    revise = all(pair["b_improvement"] <= 0 for pair in pairs) or all(pair["b_quality_advantage"] <= -QUALITY_MARGIN for pair in pairs)
    decision = "GO" if go else "REVISE" if revise else "INCONCLUSIVE"
    candidate = {"GO": "provisional detached candidate for T3 design; executable default unchanged",
                 "INCONCLUSIVE": "experimental detached candidate for T3 implementation; not validated; executable default unchanged",
                 "REVISE": None}[decision]
    return {"decision": decision,
            "pairs": pairs, "quality_margin_nats_per_byte": QUALITY_MARGIN,
            "minimum_a_anchor_differences": 16, "minimum_a_deficit_ratio": 0.10,
            "viable_model_time_estimate_supported": False,
            "scope": "T2 development diagnostic; no statistical significance or full-model forecast",
            "timing_scope": "training and child wall include diagnostics; arm ratios do not estimate incremental instrumentation overhead",
            "candidate": candidate}


def check_pair_initialization(runs: list[dict], current: dict) -> None:
    for previous in runs[:-1]:
        if previous["seed"] == current["seed"] and previous["initial_nll"] != current["initial_nll"]:
            raise ValueError("paired initial NLL mismatch")


def run_schedule(root: Path, parent: Path, directory: Path, deadline: float, report: dict) -> None:
    identity = None
    for seed, arm in SCHEDULE:
        if (parent / "STOP").exists():
            raise InterruptedError("STOP requested; campaign terminal without retry")
        if deadline - time.monotonic() < CHILD_SECONDS:
            raise TimeoutError("insufficient campaign time for next bounded child")
        if source_identity(root) != report["source_identity"]:
            raise RuntimeError("source identity changed during campaign")
        event(directory, "arm_requested", seed=seed, route_credit=arm)
        if deadline - time.monotonic() < CHILD_SECONDS:
            raise TimeoutError("insufficient campaign time at child handoff")
        child_dir, child = text_runner.execute(root, seed=seed, route_credit=arm,
                                               timeout=CHILD_SECONDS, campaign_deadline=deadline,
                                               profile="t2-credit", credit_diagnostics="on")
        report["attempted_receipts"].append(str(child_dir / "report.json"))
        if child["status"] == "stopped" or (parent / "STOP").exists():
            raise InterruptedError("owned run stopped; campaign terminal without retry")
        if child["status"] != "completed":
            raise RuntimeError(f"owned run failed: {child_dir / 'report.json'}")
        run = measurement(child_dir, child, seed, arm)
        report["runs"].append(run)
        check_pair_initialization(report["runs"], run)
        current = (run["input_sha256"], run["resolved_kernel"], run["data"], run["evaluation_examples"])
        if identity is not None and current != identity:
            raise RuntimeError("campaign binary/data/kernel/evaluation identity mismatch")
        identity = current
        if time.monotonic() >= deadline:
            raise TimeoutError("aggregate campaign deadline exceeded")
        event(directory, "arm_completed", seed=seed, route_credit=arm, receipt=run["receipt"])
        publish(directory, report)
    if source_identity(root) != report["source_identity"]:
        raise RuntimeError("source identity changed during campaign")
    if (parent / "STOP").exists():
        raise InterruptedError("STOP requested during final source verification")
    report["comparison"] = paired_decision(report["runs"])


def terminal_guard(parent: Path, started: float, report: dict) -> bool:
    if report["status"] != "completed":
        return False
    elapsed = time.monotonic() - started
    if (parent / "STOP").exists():
        report.update(status="stopped", error="STOP requested during campaign finalization")
    elif elapsed >= CAMPAIGN_SECONDS:
        report.update(status="failed", deadline_exceeded=True)
    else:
        return False
    report.pop("comparison", None)
    report["wall_seconds"] = elapsed
    return True


def finalize(directory: Path, parent: Path, started: float, report: dict) -> None:
    report["wall_seconds"] = time.monotonic() - started
    report["wall_seconds_scope"] = "sampled before terminal persistence; on correction, after first publication and before corrective write"
    if report["wall_seconds"] >= CAMPAIGN_SECONDS:
        report.update(status="failed", deadline_exceeded=True)
    terminal_guard(parent, started, report)
    if report["status"] != "completed":
        report.pop("comparison", None)
    try:
        event(directory, "campaign_finished", status=report["status"], wall_seconds=report["wall_seconds"])
    except OSError as error:
        report.update(status="failed", observer_error=str(error))
        report.pop("comparison", None)
    publish(directory, report)
    if terminal_guard(parent, started, report):
        publish(directory, report)


def execute(root: Path, *, profile: str) -> tuple[Path, dict]:
    if profile != "t2-credit":
        raise ValueError("explicit t2-credit profile required; T1 budget is consumed")
    started = time.monotonic()
    root = root.resolve(strict=True)
    parent = owned_parent(root)
    directory = parent / ("t2-campaign-" + uuid.uuid4().hex)
    directory.mkdir(exist_ok=False)
    report = {"protocol": "text-credit-campaign-v2", "profile": profile, "status": "running", "automatic_retry": False,
              "maximum_runs": 6, "deadline_seconds": CAMPAIGN_SECONDS,
              "per_child_deadline_seconds": CHILD_SECONDS, "attempted_receipts": [], "runs": [],
              "stop_file": str(parent / "STOP"), "checkpoint_resumable": False,
              "schedule": [{"seed": seed, "route_credit": arm} for seed, arm in SCHEDULE]}
    try:
        report["source_identity"] = source_identity(root)
        event(directory, "campaign_started", deadline_seconds=CAMPAIGN_SECONDS, maximum_runs=6)
        publish(directory, report)
        run_schedule(root, parent, directory, started + CAMPAIGN_SECONDS, report)
        report["status"] = "completed"
    except (KeyboardInterrupt, InterruptedError) as error:
        report.update(status="stopped", error=str(error) or "keyboard interrupt")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        finalize(directory, parent, started, report)
    return directory, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("t2-credit",))
    directory, report = execute(Path(__file__).resolve().parents[1], **vars(parser.parse_args()))
    print(f"{report['status']}: {directory / 'report.json'}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
