"""One suspended Job child and cooperative Ready stop; see causal_OWNERSHIP.md."""

import os
import subprocess
import time
from pathlib import Path

from scripts import causal_resources as resources
from scripts.causal_contract import PROTOCOL
from scripts.causal_disk import disk_bytes
from scripts.causal_observer import load, request_stop, write
from scripts.pulse_io import LOG_LIMIT


def stopped(paths: list[Path]) -> bool:
    return any(path.exists() for path in paths)


def stop_reason(paths: list[Path]) -> str | None:
    if any(path.exists() and path.name != "driver-stop.json" for path in paths):
        return "user_stop"
    for path in paths:
        if path.exists():
            value = load(path, 1024)
            if value.get("schema") != 1 or value.get("reason") not in ("user_stop", "resource_stop", "budget_stop"):
                raise ValueError("invalid external driver STOP")
            return value["reason"]
    return None


def watch(process, owner, directory: Path, stop_paths: list[Path], deadline: float,
          reserve: float, cushion: int, disk_limit: int, observer, child: dict) -> None:
    heartbeat = time.monotonic()
    grace_deadline = deadline
    while process.poll() is None:
        now = time.monotonic()
        reason = None
        if stopped(stop_paths):
            reason = stop_reason(stop_paths)
        elif now >= deadline - reserve:
            reason = "budget_stop"
        try:
            resources.sample(owner, process, child, cushion)
            if any((directory / name).stat().st_size > LOG_LIMIT for name in ("stdout.log", "stderr.log")):
                raise ValueError("causal child log detection threshold exceeded")
            if now - heartbeat >= 1:
                used = disk_bytes(directory, process.pid if process.poll() is None else None)
                child["peak_attempt_disk_bytes"] = max(used, child.get("peak_attempt_disk_bytes", 0))
                if used > disk_limit:
                    raise MemoryError("sampled attempt disk allowance exceeded")
                observer.emit("heartbeat", child=child)
                heartbeat = now
        except MemoryError as error:
            reason = reason or "resource_stop"
            child["resource_stop_detail"] = str(error)
        if reason and "stop_requested" not in child:
            request_stop(directory, reason, observer)
            child["stop_requested"] = reason
            grace_deadline = min(deadline, now + reserve)
        if now >= grace_deadline:
            raise TimeoutError("owned child did not stop before the hard/grace deadline")
        time.sleep(0.05)


def cleanup(process, job, directory: Path, owner, observer, child: dict, grace: float) -> list[str]:
    errors = []
    if process is not None and process.poll() is None:
        try:
            request_stop(directory, "failed", observer)
            process.wait(timeout=max(0.001, grace))
        except Exception as error:
            errors.append(f"cooperative cleanup: {error}")
    if job is not None:
        try:
            job.close()
            child["job_closed"] = True
        except Exception as error:
            errors.append(f"job close: {error}")
    if process is not None:
        try:
            if process.poll() is None:
                child["forced_termination"] = True
                process.kill()
            process.wait(timeout=2)
            child.update(returncode=process.returncode, reaped=process.returncode is not None)
        except Exception as error:
            child["reaped"] = False
            errors.append(f"exact handle reap: {error}")
        try:
            child["terminal_child_memory"] = resources.process_memory(process)
            child["terminal_owner_memory"] = resources.process_memory(owner)
            child["terminal_driver_memory"] = resources.process_memory(owner.driver)
            child["terminal_peak_memory_queried"] = True
        except Exception as error:
            child["terminal_peak_memory_queried"] = False
            errors.append(f"terminal memory: {error}")
    return errors


def run(root: Path, binary: Path, directory: Path, request_sha: str, nonce: str,
        owner, envelope: dict, stop_paths: list[Path], deadline: float, reserve: float,
        admission: dict, kernel: str, observer, child: dict, protocol: str = PROTOCOL) -> None:
    environment = os.environ.copy()
    environment.update({key: "1" for key in ("RAYON_NUM_THREADS", "CANDLE_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    environment.update(HEXWEAVE_KERNEL=kernel, CUDA_VISIBLE_DEVICES="",
        HEXWEAVE_CAUSAL_NONCE=nonce, HEXWEAVE_CAUSAL_REQUEST_SHA256=request_sha,
        HEXWEAVE_CAUSAL_OWNER_PID=str(owner.pid))
    command = [str(binary), "causal", "train", "--owner-request", str(directory / "worker-request.json")]
    process = job = None
    failure = None
    started = time.monotonic()
    try:
        job = resources.Job()
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, shell=False,
                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x00000004)
            child["identity"] = resources.process_identity(process)
            job.assign(process)
            launch = {"schema": 1, "protocol": protocol, "attempt_id": envelope["attempt_id"],
                      "nonce": nonce, "worker_request_sha256": request_sha, "owner": resources.process_identity(owner),
                      "child": child["identity"], "job_assigned": True, "job_active_process_limit": 1,
                      "job_commit_limit_bytes": resources.CHILD_LIMIT, "threads": 1}
            write(directory / "launch.json", launch, immutable=True)
            observer.emit("child_admitted_suspended", **launch)
            if stopped(stop_paths) or time.monotonic() >= deadline - reserve:
                raise InterruptedError("STOP or reserve reached before resume")
            resources.resume_suspended(process)
            child["resumed"] = True
            watch(process, owner, directory, stop_paths, deadline, reserve,
                  admission["host_cushion_bytes"], envelope["budget"]["disk_bytes_per_attempt"], observer, child)
            if process.returncode != 0:
                raise RuntimeError(f"causal child exited {process.returncode}")
    except KeyboardInterrupt:
        try:
            request_stop(directory, "user_stop", observer)
            child["stop_requested"] = "user_stop"
            process.wait(timeout=max(0.001, min(reserve, deadline - time.monotonic())))
            if process.returncode != 0:
                raise RuntimeError("interrupted child exited without clean checkpoint")
        except Exception as error:
            failure = error
    except BaseException as error:
        failure = error
    finally:
        errors = cleanup(process, job, directory, owner, observer, child, min(reserve, max(0, deadline - time.monotonic())))
        child["wall_seconds"] = time.monotonic() - started
        try:
            observer.emit("child_reaped" if child.get("reaped") else "child_cleanup", child=child)
        except Exception as error:
            errors.append(f"terminal observer: {error}")
        if errors:
            child["cleanup_errors"] = errors
            failure = RuntimeError("owned cleanup/evidence failed: " + "; ".join(errors))
    if failure is not None:
        raise failure
    if child.get("forced_termination") or not child.get("job_closed"):
        raise RuntimeError("child lacks clean cooperative release")
    if (child["terminal_child_memory"]["peak_working_set_bytes"] > resources.CHILD_LIMIT
            or child["terminal_owner_memory"]["peak_working_set_bytes"] > resources.OWNER_LIMIT):
        raise MemoryError("terminal historical owner/child peak exceeded sampled policy")
