"""One explicitly pinned CPU causal attempt; see causal_OWNERSHIP.md."""

import argparse
import json
import time
import uuid
from pathlib import Path

from scripts import causal_child, causal_contract, causal_prepare, causal_resources, causal_result, causal_terminal
from scripts.causal_observer import write
from scripts.pulse_io import digest


def execute(root: Path, request_path: Path, request_sha256: str) -> tuple[Path, dict]:
    started = time.monotonic()
    root = root.resolve(strict=True)
    request, paths = causal_prepare.open_request(root, request_path, request_sha256)
    context = causal_prepare.create(root, request, started)
    directory, envelope = context["directory"], context["envelope"]
    report = {"schema": 1, "protocol": request["protocol"], "status": "running", "automatic_retry": False,
              "run_id": envelope["run_id"], "attempt_id": envelope["attempt_id"], "owner_request_sha256": request_sha256,
              "immutable_identity": request["identity"], "requested_limits": request["limits"], "budget": request["budget"],
              "consumed": envelope["consumed"].copy(), "physical_work_unknown": False, "child": {}, "release_ready": False,
              "attempt_start_wall_millis": envelope["attempt_start_wall_millis"],
              "effective_deadline_millis": int((context["deadline"] - started) * 1000),
              "external_seal_required": True, "resume_authorized_by_owner_alone": False,
              "ownership_scope": "exact external driver, Python owner and one Job-contained child; adjacent samples, not instantaneous whole-tree RSS",
              "hard_limit_type": "Windows Job child commit limit; driver+owner and three-process RSS sums are sampled guards",
              "stop_files": [str(path) for path in context["stop_paths"]]}
    worker = driver = None
    try:
        write(directory / "request.json", request, immutable=True)
        write(directory / "report.json", report)
        if causal_child.stopped(context["stop_paths"]):
            raise InterruptedError("STOP requested before admission")
        owner = causal_resources.owner_process()
        driver = causal_resources.driver_process()
        owner.driver = driver
        report["admission"] = causal_resources.admission(owner, directory, request["budget"]["disk_bytes_per_attempt"])
        context["observer"].emit("physical_admission", admission=report["admission"])
        causal_contract.validate(root, request)
        worker, worker_hash = causal_prepare.worker(root, request, paths, context)
        if causal_child.stopped(context["stop_paths"]) or time.monotonic() >= context["deadline"] - request["limits"]["stop_reserve_millis"] / 1000:
            raise InterruptedError("STOP or reserve reached at child handoff")
        causal_prepare.claim(root, request, context)
        report["physical_work_unknown"] = True
        causal_child.run(root, paths["binary"], directory, worker_hash, uuid.uuid4().hex, owner,
            envelope, context["stop_paths"], context["deadline"], request["limits"]["stop_reserve_millis"] / 1000,
            report["admission"], request["identity"]["kernel"], context["observer"], report["child"], request["protocol"])
        result = causal_result.check(directory, worker, report["child"])
        accept_result(context, report, result)
    except (KeyboardInterrupt, InterruptedError) as error:
        report.update(status="user_stop", error=str(error) or "keyboard interrupt")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            if digest(paths["owner_request"]) != request_sha256:
                raise ValueError("external owner request changed")
            causal_contract.validate(root, request)
            if causal_child.disk_bytes(directory) > request["budget"]["disk_bytes_per_attempt"]:
                raise ValueError("attempt exceeded whole-directory disk allowance")
            report["immutable_inputs_rechecked"] = True
        except Exception as error:
            report.update(status="failed", identity_or_disk_error=str(error))
        causal_terminal.finish(context, report)
        if driver is not None:
            causal_resources.close_driver(driver)
    return directory, report


def accept_result(context: dict, report: dict, result: dict) -> None:
    status = report["child"].get("stop_requested", result["status"])
    if result["status"] != "completed" and status != result["status"]:
        raise ValueError("worker and owner terminal reasons disagree")
    report.update(status=status, physical_work_unknown=False,
                  result={key: result[key] for key in ("status", "checkpoint", "global_opportunity", "physical_opportunities_completed", "timings")},
                  worker_result_sha256=digest(context["directory"] / "result.json"))
    report["consumed"]["opportunities"] = result["owner"]["consumed"]["opportunities"]
    report["consumed"]["wall_millis"] = result["owner"]["consumed"]["wall_millis"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    directory, report = execute(Path(__file__).resolve().parents[1], args.request, args.request_sha256)
    print(json.dumps({"schema": 1, "protocol": report["protocol"],
          "status": report["status"], "report_path": str(directory / "report.json")}), flush=True)
    return 0 if report["release_ready"] and report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
