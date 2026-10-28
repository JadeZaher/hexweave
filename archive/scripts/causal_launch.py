"""One external owner handle and post-reap seal; see causal_OWNERSHIP.md."""

import argparse
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from scripts import causal_contract, causal_prepare, causal_resources, causal_seal
from scripts.causal_observer import contained, load, write
from scripts.pulse_io import LOG_LIMIT, digest


def stop(directory: Path, reason: str) -> None:
    path = directory / "driver-stop.json"
    if not path.exists():
        write(path, {"schema": 1, "reason": reason}, immutable=True)


def watch(process, driver, directory: Path, root: Path, deadline: float, reserve: float, report: dict, study=None) -> None:
    while process.poll() is None:
        now = time.monotonic()
        if study is not None:
            study.poll(directory, process, report)
        owner_memory = causal_resources.process_memory(process)
        driver_memory = causal_resources.process_memory(driver)
        overhead = owner_memory["working_set_bytes"] + driver_memory["working_set_bytes"]
        report["peak_sampled_driver_owner_rss"] = max(overhead, report.get("peak_sampled_driver_owner_rss", 0))
        if (root / "work/causal/STOP").exists():
            stop(directory, "user_stop")
        elif overhead > causal_resources.OWNER_LIMIT:
            stop(directory, "resource_stop")
        elif now >= deadline - reserve:
            stop(directory, "budget_stop")
        if any((directory / name).stat().st_size > LOG_LIMIT for name in ("owner.stdout", "owner.stderr")):
            raise ValueError("external owner log bound exceeded")
        if now >= deadline:
            raise TimeoutError("external owner hard wall bound exceeded")
        time.sleep(0.05)


def execute(root: Path, request_path: Path, request_sha256: str, study=None) -> tuple[Path, dict]:
    outer_started = time.monotonic()
    root = root.resolve(strict=True)
    request, _ = causal_prepare.open_request(root, request_path, request_sha256)
    if request["protocol"] != causal_contract.PROTOCOL:
        if study is None:
            raise ValueError("assay requires its conserved live study driver")
        study.admit(request, request_path, request_sha256)
    elif study is not None:
        raise ValueError("study context cannot alter v3 execution")
    parent = contained(root, "work/causal-drivers", exists=False)
    parent.mkdir(exist_ok=True, parents=True)
    directory = parent / uuid.uuid4().hex
    directory.mkdir(exist_ok=False)
    driver = causal_resources.owner_process()
    driver_identity = causal_resources.process_identity(driver)
    report = {"schema": 1, "protocol": request["protocol"], "status": "failed", "automatic_retry": False,
              "driver_identity": driver_identity, "request_sha256": request_sha256,
              "memory_scope": "outer driver/owner samples here; numerical owner separately samples all three exact processes",
              "budget_scope": "conserved owner-process wall; driver/final-seal overhead recorded separately"}
    process = None
    try:
        environment = os.environ.copy()
        environment.update(HEXWEAVE_CAUSAL_DRIVER_PID=str(driver.pid),
            HEXWEAVE_CAUSAL_DRIVER_CREATED=str(driver_identity["creation_filetime"]),
            HEXWEAVE_CAUSAL_DRIVER_STOP=str(directory / "driver-stop.json"))
        if study is not None:
            environment.update(study.environment(driver_identity, directory))
            report["study_scope"] = "shared live study driver; interval wall and live memory sample, not process-exit evidence"
        command = [sys.executable, "-B", "-m", "scripts.causal_runner", "--request", str(request_path), "--request-sha256", request_sha256]
        with (directory / "owner.stdout").open("xb") as stdout, (directory / "owner.stderr").open("xb") as stderr:
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, shell=False, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            identity = causal_resources.process_identity(process)
            report["owner_identity"] = identity
            try:
                watch(process, driver, directory, root, started + request["limits"]["wall_millis"] / 1000,
                      request["limits"]["stop_reserve_millis"] / 1000, report, study)
            except KeyboardInterrupt:
                stop(directory, "user_stop")
                process.wait(timeout=request["limits"]["stop_reserve_millis"] / 1000)
            process.wait(timeout=0)
            wall = math.ceil((time.monotonic() - started) * 1000)
            report.update(owner_exit_code=process.returncode, owner_reaped=True, external_owner_wall_millis=wall)
            report["terminal_owner_memory"] = causal_resources.process_memory(process)
            if study is not None:
                study.poll(directory, process, report)
        if process.returncode != 0:
            raise RuntimeError("owner did not finish a clean release")
        result = load(directory / "owner.stdout", 4096)
        attempt = contained(root / "work/causal", str(Path(result["report_path"]).parent))
        proof = causal_seal.seal(root, attempt, process, identity, wall, driver_identity)
        report.update(status=result["status"], attempt_directory=str(attempt), release_seal_sha256=digest(attempt / "release-seal.json"), release_seal=proof)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        cleanup(process, directory, report)
        report["terminal_driver_memory"] = causal_resources.process_memory(driver)
        report["driver_wall_millis_before_final_receipt"] = math.ceil((time.monotonic() - outer_started) * 1000)
        report["wall_scope"] = "driver elapsed sampled before its final receipt write; caller must measure complete external process wall"
        if study is not None:
            try:
                study.poll(directory, process, report, force=True)
            except Exception as error:
                report.update(status="failed", study_finalization_error=str(error))
                study.stopped = "failed"
            study.revoke(report)
        write(directory / "driver-report.json", report)
    return directory, report


def cleanup(process, directory: Path, report: dict) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        stop(directory, "user_stop")
        process.wait(timeout=2)
    except Exception:
        process.kill()
        process.wait(timeout=2)
        report["forced_owner_termination"] = True
    report.update(owner_reaped=process.poll() is not None, owner_exit_code=process.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    directory, report = execute(Path(__file__).resolve().parents[1], args.request.resolve(strict=True), args.request_sha256)
    print(f"{report['status']}: {directory / 'driver-report.json'}", flush=True)
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
