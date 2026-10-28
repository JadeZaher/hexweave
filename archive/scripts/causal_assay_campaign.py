"""Execute one explicitly authorized conserved assay; never clears program STOP."""

import argparse
import time
from pathlib import Path

from scripts import causal_assay as assay, causal_assay_study as study, causal_contract, causal_launch
from scripts.causal_assay_guard import Guard
from scripts.causal_observer import load, write


def diagnostics(guard: Guard, report: dict) -> None:
    from scripts.causal_assay_score import arm
    report["diagnostics"] = []
    for item in guard.records:
        guard.poll(None, None, {}, force=True)
        if guard.stopped:
            break
        if item.get("attempt_directory"):
            report["diagnostics"].append(arm(Path(item["attempt_directory"])))
        guard.poll(None, None, {}, force=True)
        if guard.stopped:
            break
    if guard.stopped:
        report["status"] = guard.stopped
    report["every_seed_both_arms_gate"] = (report["status"] == "completed"
        and len(report["diagnostics"]) == 6 and all(item["diagnostic_gate"] for item in report["diagnostics"]))
    report["every_seed_gate_by_rule"] = {rule: report["status"] == "completed"
        and len([item for item in report["diagnostics"] if item["rule"] == rule]) == 3
        and all(item["diagnostic_gate"] for item in report["diagnostics"] if item["rule"] == rule)
        for rule in ("dense", "magnitude")}


def execute(root: Path, plan_path: Path, plan_sha256: str) -> tuple[Path, dict]:
    started = time.monotonic()
    root = root.resolve(strict=True)
    path = causal_contract.pin(root, {"path": str(plan_path), "sha256": plan_sha256}, 16_384)
    plan = load(path, 16_384)
    if (set(plan) != {"schema", "protocol", "study", "requests", "training_authorized"}
            or plan["schema"] != 1 or plan["protocol"] != assay.PROTOCOL
            or plan["training_authorized"] is not False or len(plan["requests"]) != 6):
        raise ValueError("invalid six-arm preparation plan")
    directory, declaration = study.declaration(root, plan["study"])
    if path != directory / "plan.json":
        raise ValueError("noncanonical study plan")
    for index, spec in enumerate(plan["requests"]):
        request_path = causal_contract.pin(root, spec, 65_536)
        if request_path != directory / f"arm-{index}.json":
            raise ValueError("noncanonical study arm request")
        request = load(request_path, 65_536)
        causal_contract.validate(root, request)
        if request["operation"]["arm"] != index or request["operation"]["study"] != plan["study"]:
            raise ValueError("reordered study arm")
    guard = Guard(root, directory, plan, started)
    report = {"schema": 1, "protocol": assay.PROTOCOL, "status": "failed", "plan_sha256": plan_sha256,
              "study_sha256": plan["study"]["sha256"], "driver_identity": guard.identity,
              "automatic_retry": False, "resume_campaign": False, "useful_assistant_claim": False}
    try:
        for spec in plan["requests"]:
            arm_directory, arm = causal_launch.execute(root, Path(spec["path"]), spec["sha256"], study=guard)
            guard.record(arm_directory, arm)
            if guard.stopped:
                break
        guard.poll(None, None, {}, force=True)
        report["status"] = guard.stopped or ("completed" if len(guard.records) == 6 else "failed")
        diagnostics(guard, report)
    except (KeyboardInterrupt, InterruptedError) as error:
        guard.stopped = guard.stopped or "user_stop"
        report.update(status=guard.stopped or "user_stop", error=str(error))
    except Exception as error:
        guard.stopped = guard.stopped or "failed"
        report.update(status=guard.stopped or "failed", error=f"{type(error).__name__}: {error}")
    finally:
        report.update(arms=guard.records, elapsed_millis_before_final_persistence=int((time.monotonic() - guard.started) * 1000),
            driver_scope="one still-live sampled study driver; absence must be checked after whole process exit",
            owner_wall_ceiling_millis_sum=6 * study.LIMITS["wall_millis"], whole_wall_ceiling_millis=study.WHOLE_MILLIS)
        guard.observer.emit("study_terminal", status=report["status"], arms=len(guard.records))
        write(directory / "report.json", report)
        try:
            guard.poll(None, None, {}, force=True)
        except Exception as error:
            guard.stopped = guard.stopped or "failed"
            report["finalization_error"] = str(error)
        if guard.stopped:
            if report["status"] == "completed":
                report["status"] = guard.stopped
            guard.revoke_last()
            report["every_seed_both_arms_gate"] = False
            report["every_seed_gate_by_rule"] = {"dense": False, "magnitude": False}
        report["terminal_owner_free_resources"] = guard.owner_free_sample
        write(directory / "report.json", report)
        stop = root / "work/causal/STOP"
        if not stop.exists():
            write(stop, {"schema": 1, "study": declaration["run_id"], "reason": "terminal_study_disposition",
                         "status": report["status"]}, immutable=True)
    return directory, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    directory, report = execute(Path(__file__).resolve().parents[1], args.plan, args.plan_sha256)
    print(f"{report['status']}: {directory / 'report.json'}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
