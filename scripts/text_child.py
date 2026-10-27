"""Own one suspended child, assign hard limits, then monitor cooperatively."""

import os
import subprocess
import time
from pathlib import Path

from scripts.pulse_io import LOG_LIMIT, event
from scripts.text_resources import CHILD_LIMIT, Job, memory_status, process_memory, resume_suspended


def stop_owned(process, text_directory: Path) -> None:
    if process.poll() is not None:
        return
    if text_directory.is_dir():
        with (text_directory / "STOP").open("a", encoding="utf-8") as stream:
            stream.write("supervisor stop\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            pass
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def watch(process, directory: Path, parent: Path, logs: tuple[Path, Path],
          deadline: float, cushion: int, child: dict) -> None:
    heartbeat = time.monotonic()
    while process.poll() is None:
        now = time.monotonic()
        if any(path.exists() for path in (parent / "STOP", directory / "STOP", directory / "text/STOP")):
            raise InterruptedError("STOP requested; terminal without retry")
        if now >= deadline:
            raise TimeoutError("bounded text deadline exceeded")
        if any(path.stat().st_size > LOG_LIMIT for path in logs):
            raise ValueError("text log detection threshold exceeded")
        memory = process_memory(process)
        record_memory(child, memory)
        if memory["working_set_bytes"] > CHILD_LIMIT:
            raise MemoryError("sampled working-set guard exceeded 512 MiB")
        if memory_status()["available_physical_bytes"] < cushion:
            raise MemoryError("adaptive host physical-memory cushion reached")
        if now - heartbeat >= 1:
            event(directory, "heartbeat", pid=process.pid, memory=memory)
            heartbeat = now
        time.sleep(0.05)


def run_child(root: Path, command: list[str], directory: Path, parent: Path,
              deadline: float, cushion: int, backend: str, child: dict) -> None:
    logs = (directory / "stdout.json", directory / "stderr.log")
    environment = os.environ.copy()
    environment.update({name: "1" for name in (
        "RAYON_NUM_THREADS", "CANDLE_NUM_THREADS", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    environment["HEXWEAVE_KERNEL"] = backend
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = None
    job = None
    failure = None
    started = time.monotonic()
    try:
        job = Job()
        with logs[0].open("xb") as stdout, logs[1].open("xb") as stderr:
            process = subprocess.Popen(command, cwd=root, env=environment,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, shell=False,
                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x00000004)
            child["pid"] = process.pid
            job.assign(process)
            event(directory, "child_admitted_suspended", pid=process.pid,
                  commit_limit_bytes=CHILD_LIMIT, maximum_processes=1)
            resume_suspended(process)
            child["resumed"] = True
            watch(process, directory, parent, logs, deadline, cushion, child)
            if process.returncode != 0:
                raise RuntimeError(f"text child failed with exit {process.returncode}")
    except BaseException as error:
        failure = error
    finally:
        errors = cleanup(process, job, directory / "text", child)
        child["wall_seconds"] = time.monotonic() - started
        try:
            event(directory, "child_reaped" if child.get("reaped") else "child_cleanup", **child)
        except OSError as error:
            errors.append(f"observer: {error}")
        if errors:
            child["cleanup_errors"] = errors
            if failure is None or child.get("reaped") is False:
                failure = RuntimeError("owned child cleanup failed: " + "; ".join(errors))
    if failure is not None:
        raise failure
    if any(path.stat().st_size > LOG_LIMIT for path in logs):
        raise ValueError("text log detection threshold exceeded")
    if child["memory"]["peak_working_set_bytes"] > CHILD_LIMIT:
        raise MemoryError("historical peak working-set guard exceeded 512 MiB")


def cleanup(process, job, text_directory: Path, child: dict) -> list[str]:
    errors = []
    if process is not None:
        try:
            stop_owned(process, text_directory)
        except Exception as error:
            errors.append(f"cooperative stop: {error}")
    if job is not None:
        try:
            job.close()
        except Exception as error:
            errors.append(f"job close: {error}")
    if process is not None:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
            child["returncode"] = process.returncode
            child["reaped"] = process.returncode is not None
        except Exception as error:
            errors.append(f"owned handle reap: {error}")
            child["reaped"] = False
        try:
            record_memory(child, process_memory(process))
            child["terminal_peak_memory_queried"] = True
        except Exception as error:
            errors.append(f"terminal peak memory: {error}")
            child["terminal_peak_memory_queried"] = False
    return errors


def record_memory(child: dict, sample: dict) -> None:
    previous = child.get("memory", {})
    child["memory"] = sample | {key: max(sample.get(key, 0), previous.get(key, 0))
                               for key in ("peak_working_set_bytes", "peak_pagefile_bytes")}
