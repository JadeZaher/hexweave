"""Run finite independent pulse fixtures and stop on the first failure."""

import argparse
import os
import subprocess
import time
import uuid
from pathlib import Path

from scripts.pulse_io import LOG_LIMIT, digest, event, owned_directory, publish, read_result


def validate(steps: int, seeds: list[int], timeout: float, residents: int,
             anchor_interval: int, mode: str) -> None:
    if not 1 <= steps <= 256 or not 1 <= residents <= 3:
        raise ValueError("steps must be 1..256 and residents 1..3")
    if not 1 <= anchor_interval <= 16 or not 1 <= timeout <= 120:
        raise ValueError("anchor interval must be 1..16 and timeout 1..120 seconds")
    if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
        raise ValueError("provide 1..3 unique independent seeds")
    if any(type(seed) is not int or not 0 <= seed <= 2**64 - 1 for seed in seeds):
        raise ValueError("seeds must be unsigned 64-bit integers")
    if mode not in ("pulse", "anchor-only"):
        raise ValueError("unsupported pulse mode")


def stop_owned(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def watch(process, directory: Path, stop: Path, logs: tuple[Path, Path],
          timeout: float) -> None:
    started = time.monotonic()
    heartbeat = started
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if stop.exists() or (directory / "STOP").exists():
            raise InterruptedError("STOP requested; no automatic resume")
        if elapsed >= timeout:
            raise TimeoutError("bounded child deadline exceeded")
        if any(path.stat().st_size > LOG_LIMIT for path in logs):
            raise ValueError("bounded child log limit exceeded")
        if time.monotonic() - heartbeat >= 1:
            event(directory, "heartbeat", pid=process.pid, elapsed_seconds=elapsed)
            heartbeat = time.monotonic()
        time.sleep(0.05)


def run_child(command: list[str], directory: Path, stop: Path, timeout: float,
              seed: int, steps: int, mode: str, residents: int,
              anchor_interval: int) -> dict:
    logs = (directory / f"seed-{seed}.stdout.json", directory / f"seed-{seed}.stderr.log")
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    process = None
    started = time.monotonic()
    with logs[0].open("xb") as stdout, logs[1].open("xb") as stderr:
        try:
            process = subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, shell=False,
                                       creationflags=flags)
            event(directory, "child_started", pid=process.pid, seed=seed, command=command)
            watch(process, directory, stop, logs, timeout - (time.monotonic() - started))
            if process.returncode != 0:
                raise RuntimeError(f"child failed with exit {process.returncode}; see {logs[1].name}")
        finally:
            if process is not None:
                stop_owned(process)
                event(directory, "child_exited", pid=process.pid, seed=seed,
                      returncode=process.returncode)
    if any(path.stat().st_size > LOG_LIMIT for path in logs):
        raise ValueError("bounded child log limit exceeded")
    return {"seed": seed, "wall_seconds": time.monotonic() - started,
            "result": read_result(logs[0], steps, seed, mode, residents, anchor_interval)}


def execute(root: Path, *, steps: int = 64, seeds: list[int] | None = None,
            timeout: float = 30, residents: int = 1, anchor_interval: int = 4,
            mode: str = "pulse", profile: str = "debug") -> tuple[Path, dict]:
    seeds = [7] if seeds is None else seeds
    validate(steps, seeds, timeout, residents, anchor_interval, mode)
    if profile not in ("debug", "release"):
        raise ValueError("unsupported build profile")
    root = root.resolve(strict=True)
    suffix = ".exe" if os.name == "nt" else ""
    executable = (root / f"work/target/{profile}/hexweave{suffix}").resolve(strict=True)
    if not executable.is_relative_to(root) or not executable.is_file():
        raise ValueError("executable must be built inside this repository")
    binary_hash = digest(executable)
    parent = owned_directory(root)
    directory = parent / uuid.uuid4().hex
    directory.mkdir(exist_ok=False)
    report = {"status": "running", "steps_per_seed": steps, "seeds": seeds,
              "independent_initializations": True, "resume": False,
              "binary_sha256": binary_hash, "results": [],
              "native_chat_wakeup": False, "checkpoint": None,
              "deadline_seconds": timeout, "log_detection_threshold_bytes": LOG_LIMIT}
    deadline = time.monotonic() + timeout
    try:
        event(directory, "run_started", **report)
        for seed in seeds:
            if (parent / "STOP").exists() or (directory / "STOP").exists():
                raise InterruptedError("STOP requested before launch")
            if digest(executable) != binary_hash:
                raise RuntimeError("executable changed during the experiment")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("whole-run deadline exceeded before launch")
            command = [str(executable), "pulse", "--steps", str(steps), "--seed", str(seed),
                       "--mode", mode, "--residents", str(residents),
                       "--anchor-interval", str(anchor_interval)]
            report["results"].append(run_child(command, directory, parent / "STOP",
                                               remaining, seed, steps, mode,
                                               residents, anchor_interval))
        report["status"] = "completed"
    except (KeyboardInterrupt, InterruptedError) as error:
        report.update(status="stopped", error=str(error) or "keyboard interrupt")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            event(directory, "run_finished", status=report["status"])
        except OSError as error:
            report.update(status="failed", observer_error=str(error))
        publish(directory, report)
    return directory, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--residents", type=int, default=1)
    parser.add_argument("--anchor-interval", type=int, default=4)
    parser.add_argument("--mode", choices=("pulse", "anchor-only"), default="pulse")
    parser.add_argument("--profile", choices=("debug", "release"), default="debug")
    args = vars(parser.parse_args())
    directory, report = execute(Path(__file__).resolve().parents[1], **args)
    print(f"{report['status']}: {directory / 'report.json'}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
