"""Finite terminal publication and explicit resume release; see causal_OWNERSHIP.md."""

import math
import time

from scripts.causal_child import stop_reason
from scripts.causal_observer import load, write
from scripts.pulse_io import digest


PERSISTENCE_ALLOWANCE_MILLIS = 1000


def elapsed(context: dict) -> int:
    return math.ceil((time.monotonic() - context["started"]) * 1000)


def guard(context: dict, report: dict) -> bool:
    old = report["status"]
    if old == "failed":
        return False
    if time.monotonic() >= context["deadline"]:
        report.update(status="failed", deadline_exceeded=True, error="owner whole-attempt deadline exceeded")
    elif reason := stop_reason(context["stop_paths"]):
        report["status"] = reason
    return report["status"] != old


def invalidate(context: dict, report: dict, reason: str) -> None:
    directory = context["directory"]
    report.update(status="failed", release_ready=False, release_error=reason)
    path = directory / "release-invalid.json"
    if not path.exists():
        write(path, {"schema": 1, "reason": reason}, immutable=True)
    write(directory / "report.json", report)


def release(context: dict, report: dict) -> None:
    directory, observer, envelope = context["directory"], context["observer"], context["envelope"]
    if report["status"] == "failed" or report["physical_work_unknown"] or "result" not in report:
        return
    charged = envelope["attempt_start_wall_millis"] + elapsed(context) + PERSISTENCE_ALLOWANCE_MILLIS
    if charged >= envelope["budget"]["maximum_wall_millis"]:
        report["status"] = "budget_stop"
    report["consumed"]["wall_millis"] = charged
    acknowledgement = load(directory / "observer.json", 1024)
    if acknowledgement["sequence"] != observer.sequence or acknowledgement["event_sha256"] != observer.previous:
        raise ValueError("terminal observer acknowledgement changed")
    value = {"schema": 1, "run_id": envelope["run_id"], "attempt_id": envelope["attempt_id"],
             "owner_identity": report["admission"]["owner"],
             "checkpoint": report["result"]["checkpoint"], "status": report["status"], "consumed": report["consumed"],
             "observer_sequence": observer.sequence, "child_reaped": True,
             "observer_acknowledged": True, "physical_work_unknown": False}
    write(directory / "release.json", value, immutable=True)
    released_hash = digest(directory / "release.json")
    write(directory / "release-ready.json", {"schema": 1, "release_sha256": released_hash}, immutable=True)
    report.update(release_ready=True, release_sha256=released_hash)
    write(directory / "report.json", report)
    actual_charge = envelope["attempt_start_wall_millis"] + elapsed(context)
    changed = guard(context, report)
    if changed or actual_charge > charged:
        invalidate(context, report, "terminal STOP/deadline or publication exceeded charged allowance")


def finish(context: dict, report: dict) -> None:
    envelope, directory = context["envelope"], context["directory"]
    report["owner_wall_millis_before_terminal_persistence"] = elapsed(context)
    report["wall_scope"] = "owner elapsed sampled before terminal persistence; clean release charges an extra fixed publication allowance; external process wall separate"
    report["terminal_persistence_allowance_millis"] = PERSISTENCE_ALLOWANCE_MILLIS
    report["consumed"]["wall_millis"] = max(report["consumed"]["wall_millis"], envelope["attempt_start_wall_millis"] + elapsed(context))
    guard(context, report)
    try:
        context["observer"].emit("attempt_finished", status=report["status"], consumed=report["consumed"],
                                 physical_work_unknown=report["physical_work_unknown"])
        write(directory / "report.json", report)
        if guard(context, report):
            write(directory / "report.json", report)
        release(context, report)
    except Exception as error:
        if (directory / "release.json").exists():
            invalidate(context, report, f"terminal publication failed: {error}")
        else:
            report.update(status="failed", release_ready=False, observer_error=str(error))
            write(directory / "report.json", report)
