"""Own one core training run (`hexweave train`): Job-capped child, elastic floor, graceful STOP, receipts."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path, PurePath

from scripts.owner_resources import CHILD_LIMIT, Job, memory_status, process_memory, resume_suspended

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 256 * 1024**2
START_FLOOR = FLOOR + CHILD_LIMIT
FLOOR_SAMPLES = 10
GRACE_SECONDS = 180
BACKSTOP_SECONDS = 1800
CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return {"head": head.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def archive(path: Path, label: str) -> Path:
    """Rename `path` to a unique `<name>.<label>-<stamp>[-n]` sibling."""
    base = f"{path.name}.{label}-{stamp()}"
    target, index = path.with_name(base), 1
    while target.exists():
        target, index = path.with_name(f"{base}-{index}"), index + 1
    path.rename(target)
    return target


def run_directory(config: dict) -> Path:
    """The run directory; only work/core-runs/<name> is accepted, matching the trainer."""
    parts = PurePath(config["out"]).parts
    if len(parts) != 3 or parts[:2] != ("work", "core-runs") or parts[2] in (".", ".."):
        raise SystemExit("out must be work/core-runs/<name>")
    return ROOT.joinpath(*parts)


def lock(out: Path) -> Path:
    """Exclusive single-owner lock; one left by a crashed owner must be inspected and removed by hand."""
    path = out / "OWNER.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{path} exists: another owner holds this run, or a crashed one left it") from None
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        file.write(f"pid {os.getpid()} {stamp()}\n")
    return path


def prepare(config: dict, out: Path, clear_stop: bool) -> Path:
    """Refuse unsafe launches, then pin the binary and set aside a stale STOP and report."""
    checkpoints, binary, stop = out / "checkpoints", out / "bin" / "hexweave.exe", out / "STOP"
    if not config["resume"]:
        for held in (checkpoints, out / "best"):
            if held.exists() and any(held.iterdir()):
                raise SystemExit(f"{held} is not empty; set resume or choose a new out")
        if binary.exists():
            raise SystemExit(f"{binary} is pinned by an earlier launch; set resume or choose a new out")
    if stop.exists() and not clear_stop:
        if not stop.read_text(encoding="utf-8", errors="replace").startswith("owner:"):
            raise SystemExit(f"{stop} was not written by an owner; pass --clear-stop to archive it")
    if not binary.exists():
        binary.parent.mkdir(parents=True, exist_ok=True)
        partial = binary.with_name(f"{binary.name}.partial-{os.getpid()}")
        try:
            shutil.copy2(ROOT / "work/target/release/hexweave.exe", partial)
            os.replace(partial, binary)
        finally:
            partial.unlink(missing_ok=True)
    if stop.exists():
        archive(stop, "archived")
    if (out / "report.json").exists():
        archive(out / "report.json", "previous")
    return binary


def request_stop(out: Path, reason: str) -> float:
    (out / "STOP").write_text(f"owner: {reason}\n", encoding="utf-8")
    return time.monotonic()


def followed_by(earlier: str | None, later: str) -> str:
    """Record an overriding reason without losing the stop that preceded it."""
    return later if earlier is None else f"{earlier} then {later}"


def supervise(process, out: Path, wall_seconds: int) -> tuple[str | None, int]:
    """Sample each second; a sustained floor breach, overrun or Ctrl+C requests STOP, then kills after grace."""
    started, requested, reason, peak, breaches = time.monotonic(), None, None, 0, 0
    try:
        while process.poll() is None:
            peak = max(peak, process_memory(process)["peak_working_set_bytes"])
            breaches = breaches + 1 if memory_status()["available_physical_bytes"] < FLOOR else 0
            overrun = time.monotonic() - started > wall_seconds + BACKSTOP_SECONDS
            if requested is None and (breaches >= FLOOR_SAMPLES or overrun):
                reason = "owner_overrun_stop" if overrun else "owner_memory_stop"
                requested = request_stop(out, reason)
            if requested is not None and time.monotonic() - requested > GRACE_SECONDS:
                process.kill()
                return followed_by(reason, "owner_killed_after_grace"), peak
            time.sleep(1)
    except KeyboardInterrupt:
        reason = followed_by(reason, "owner_interrupted")
        request_stop(out, "owner_interrupted")
        try:
            process.wait(timeout=GRACE_SECONDS)
            return reason, peak
        except subprocess.TimeoutExpired:
            process.kill()
            return followed_by(reason, "owner_killed_after_grace"), peak
    return reason, peak


def stop_reason(out: Path) -> str | None:
    """Reason in an owner-written STOP (this launch's; prepare archived older ones), if present."""
    try:
        text = (out / "STOP").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text.removeprefix("owner: ") if text.startswith("owner:") else None


def settle(process, out: Path, reason: str) -> bool:
    """After an owner error, give a live child STOP and the grace period; False if it outlived the grace."""
    if process.poll() is not None:
        return True
    if not (out / "STOP").exists():
        request_stop(out, reason)
    try:
        process.wait(timeout=GRACE_SECONDS)
        return True
    except subprocess.TimeoutExpired:
        return False


def final_peak(process, peak: int) -> int:
    """Peak working set after exit (the Popen handle stays open); the polled peak if unavailable."""
    try:
        return max(peak, process_memory(process)["peak_working_set_bytes"]) if process else peak
    except OSError:
        return peak


def read_report(path: Path) -> dict | None:
    """This launch's report.json; an unreadable or malformed one is recorded, never fatal to the receipt."""
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"unparseable": True}
    return report if isinstance(report, dict) else {"unparseable": True}


def finish(out: Path, run_stamp: str, process, reason: str | None, peak: int, started: float) -> None:
    """End receipt, embedding this launch's report (a stale one was archived before launch)."""
    report = read_report(out / "report.json")
    returncode = process.returncode if process is not None else None
    fallback = (report or {}).get("terminal") or ("exited" if returncode == 0 else "failed")
    write(out / f"owner-end-{run_stamp}.json", {
        "schema": 1, "returncode": returncode, "reason": reason or fallback,
        "peak_working_set_bytes": peak, "wall_seconds": time.monotonic() - started,
        "ended_utc": stamp(), "report": report})


def own(config_path: Path, config: dict, out: Path, binary: Path, host: dict) -> int:
    run_stamp = stamp()
    write(out / f"owner-start-{run_stamp}.json", {
        "schema": 1, "config": str(config_path), "config_sha256": digest(config_path),
        "binary_sha256": digest(binary), "git": git_state(), "host": host,
        "child_commit_limit_bytes": CHILD_LIMIT, "floor_bytes": FLOOR, "started_utc": run_stamp})
    environment = os.environ.copy()
    environment.update({name: "1" for name in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    environment["CUDA_VISIBLE_DEVICES"] = ""
    job, process, started, reason, peak = None, None, time.monotonic(), None, 0
    try:
        job = Job()
        with (out / "stdout.log").open("ab") as stdout, (out / "stderr.log").open("ab") as stderr:
            process = subprocess.Popen(
                [str(binary), "train", "--config", str(config_path)], cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, shell=False,
                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | CREATE_SUSPENDED
                | CREATE_NEW_PROCESS_GROUP)
            job.assign(process)
            resume_suspended(process)
            try:
                reason, peak = supervise(process, out, int(config["wall_seconds"]))
            except Exception as error:
                reason = followed_by(stop_reason(out), f"owner_error: {type(error).__name__}")
                if not settle(process, out, reason):
                    reason = followed_by(reason, "owner_killed_after_grace")
                raise
            process.wait(timeout=60)
    except BaseException as error:
        reason = reason or followed_by(stop_reason(out), f"owner_error: {type(error).__name__}")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if job is not None:
            job.close()
        finish(out, run_stamp, process, reason, final_peak(process, peak), started)
    return 0 if process is not None and process.returncode == 0 else 1


def launch(config_path: Path, clear_stop: bool) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = run_directory(config)
    host = memory_status()
    if host["available_physical_bytes"] < START_FLOOR:
        raise SystemExit(f"available {host['available_physical_bytes']} below start floor {START_FLOOR}")
    out.mkdir(parents=True, exist_ok=True)
    held = lock(out)
    try:
        return own(config_path, config, out, prepare(config, out, clear_stop), host)
    finally:
        held.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clear-stop", action="store_true", help="archive a STOP no owner wrote")
    arguments = parser.parse_args()
    return launch(arguments.config.resolve(strict=True), arguments.clear_stop)


if __name__ == "__main__":
    raise SystemExit(main())
