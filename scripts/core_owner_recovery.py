"""Bounded memory-stop recovery with one persistent owner lock; see scripts/AGENTS.md."""

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from scripts import core_owner as owner
from scripts import core_owner_checkpoint as check

STABLE_SAMPLES = 30


@dataclass(frozen=True)
class Policy:
    max_restarts: int = 3
    wait_seconds: int = 21600

    def validate(self, config: dict) -> None:
        if not 0 <= self.max_restarts <= 3 or not 1 <= self.wait_seconds <= 21600:
            raise SystemExit("memory recovery permits 0..3 restarts and 1..21600 wait seconds")
        if config.get("connected") is None:
            raise SystemExit("--resume-on-memory currently supports connected runs only")
        if type(config.get("wall_seconds")) is not int or config["wall_seconds"] < 1:
            raise SystemExit("memory recovery requires a positive integer wall_seconds budget")


class Cancelled(Exception):
    pass


class MemoryUnavailable(Exception):
    pass


def stop_snapshot(path: Path):
    try:
        with path.open("rb") as file:
            before = os.fstat(file.fileno())
            raw = file.read(4097)
            after = os.fstat(file.fileno())
        current = path.stat()
    except FileNotFoundError:
        return None
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if len(raw) > 4096 or identity(before) != identity(after) or identity(after) != identity(current):
        raise Cancelled("STOP changed while reading")
    return (*identity(after), hashlib.sha256(raw).hexdigest()), raw


def owned_stop(out: Path, token: str | None):
    snapshot = stop_snapshot(out / "STOP")
    expected = ["owner: owner_memory_stop"] + ([f"attempt: {token}"] if token else [])
    if snapshot is None or snapshot[1].decode("utf-8", errors="replace").splitlines() != expected:
        raise Cancelled("STOP is missing or was replaced by the operator")
    return snapshot


def archive_owned_stop(out: Path, expected) -> None:
    stop = out / "STOP"
    if stop_snapshot(stop) != expected:
        raise Cancelled("STOP changed before recovery")
    archived = owner.archive(stop, "recovered")
    if stop_snapshot(archived) != expected:
        try:
            with stop.open("xb") as file:
                file.write(archived.read_bytes())
        except FileExistsError:
            pass
        raise Cancelled("STOP replacement was preserved; recovery cancelled")
    if stop.exists():
        raise Cancelled("operator STOP arrived during recovery")


def publish(out: Path, session: str, state: dict, phase: str, **updates) -> None:
    state.update(updates, phase=phase, updated_utc=owner.stamp())
    path = out / f"owner-recovery-{session}.json"
    partial = path.with_suffix(".json.partial")
    owner.write(partial, state)
    os.replace(partial, path)


def wait_ready(out: Path, expected, budget: float, sample) -> tuple[dict | None, float]:
    started, stable = time.monotonic(), 0
    while time.monotonic() - started < budget:
        if stop_snapshot(out / "STOP") != expected:
            raise Cancelled("operator STOP changed during recovery wait")
        remaining = budget - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(1, remaining))
        if stop_snapshot(out / "STOP") != expected:
            raise Cancelled("operator STOP changed during recovery wait")
        host = owner.memory_status()
        if time.monotonic() - started >= budget:
            break
        stable = stable + 1 if host["available_physical_bytes"] >= owner.START_FLOOR else 0
        elapsed = time.monotonic() - started
        sample(host, elapsed, stable)
        if stable >= STABLE_SAMPLES:
            return host, elapsed
    return None, time.monotonic() - started


def attempt(out: Path, session: str, index: int, config: dict, binary: Path,
            pinned: str, expected) -> tuple[int, dict, float]:
    if stop_snapshot(out / "STOP") != expected:
        raise Cancelled("operator STOP arrived before launch")
    if check.sha(binary) != pinned:
        raise ValueError("pinned binary changed during recovery")
    host = owner.memory_status()
    if host["available_physical_bytes"] < owner.START_FLOOR:
        raise MemoryUnavailable
    if expected is not None:
        archive_owned_stop(out, expected)
    if (out / "STOP").exists():
        raise Cancelled("operator STOP arrived before launch")
    if (out / "report.json").exists():
        owner.archive(out / "report.json", "previous")
    tag = f"{session}-{index:03d}"
    path = out / f"owner-attempt-{tag}.json"
    with path.open("x", encoding="utf-8") as file:
        json.dump(config, file, indent=1)
        file.flush()
        os.fsync(file.fileno())
    if (out / "STOP").exists():
        raise Cancelled("operator STOP arrived before launch")
    started = time.monotonic()
    code = owner.own(path, config, out, binary, host, attempt=tag)
    return code, check.read(out / f"owner-end-{tag}.json"), time.monotonic() - started


def run(config_path: Path, config: dict, out: Path, clear_stop: bool, policy: Policy) -> int:
    session = f"{owner.stamp()}-{uuid.uuid4().hex[:8]}"
    state = {"schema": 1, "config": str(config_path), "initial_wall_seconds": config["wall_seconds"],
             "remaining_wall_seconds": config["wall_seconds"], "wait_seconds": 0.0,
             "attempts": 0, "restarts": 0, "max_restarts": policy.max_restarts,
             "max_wait_seconds": policy.wait_seconds, "stable_samples_required": STABLE_SAMPLES}
    publish(out, session, state, "admitting")
    try:
        return drive(out, session, state, config, clear_stop, policy)
    except Cancelled as error:
        publish(out, session, state, "cancelled", reason=str(error))
        return 0
    except KeyboardInterrupt:
        owner.request_stop(out, "owner_interrupted")
        publish(out, session, state, "cancelled", reason="owner_interrupted")
        return 130
    except BaseException as error:
        publish(out, session, state, "failed", reason=f"{type(error).__name__}: {error}")
        raise


def drive(out: Path, session: str, state: dict, config: dict, clear_stop: bool, policy: Policy) -> int:
    binary, expected, pending = out / "bin/hexweave.exe", None, None
    need_wait = config["resume"]
    if config["resume"]:
        receipt = check.latest_end(out)
        admitted = check.inspect(config, out, binary, receipt)
        previous, token, _ = admitted
        pending = receipt, admitted
        expected = owned_stop(out, token)
        if previous["anchors"] >= config["anchors"]:
            publish(out, session, state, "finished", reason="target_complete")
            return 0
    else:
        binary = owner.prepare(config, out, clear_stop, preserve_stop=True)
    pinned = check.sha(binary)
    while state["remaining_wall_seconds"] >= 1:
        if need_wait or owner.memory_status()["available_physical_bytes"] < owner.START_FLOOR:
            waited = state["wait_seconds"]
            def sample(host, elapsed, stable):
                publish(out, session, state, "waiting_for_memory", wait_seconds=waited + elapsed,
                        available_physical_bytes=host["available_physical_bytes"], stable_samples=stable)
            host, elapsed = wait_ready(out, expected, policy.wait_seconds - waited, sample)
            state["wait_seconds"] = waited + elapsed
            if host is None:
                publish(out, session, state, "finished", reason="recovery_wait_exhausted")
                return 0
            need_wait = False
        if pending is not None and check.inspect(config, out, binary, pending[0]) != pending[1]:
            raise ValueError("recovery checkpoint or receipt changed during the wait")
        next_config = dict(config, resume=config["resume"] or state["attempts"] > 0,
                           wall_seconds=int(state["remaining_wall_seconds"]))
        publish(out, session, state, "running", attempts=state["attempts"] + 1)
        try:
            code, end, elapsed = attempt(out, session, state["attempts"], next_config, binary, pinned, expected)
        except MemoryUnavailable:
            state["attempts"] -= 1
            need_wait = True
            continue
        expected = None
        state["remaining_wall_seconds"] = max(0.0, state["remaining_wall_seconds"] - elapsed)
        report = end.get("report") or {}
        if (code != 0 or end.get("returncode") != 0 or end.get("reason") != "owner_memory_stop"
                or report.get("terminal") != "stopped" or report.get("anchors", 0) >= config["anchors"]):
            publish(out, session, state, "finished", reason=end.get("reason"), returncode=code)
            return code
        if state["remaining_wall_seconds"] < 1 or state["restarts"] >= policy.max_restarts:
            break
        receipt = out / f"owner-end-{session}-{state['attempts']:03d}.json"
        admitted = check.inspect(config, out, binary, receipt)
        _, token, _ = admitted
        pending = receipt, admitted
        expected = owned_stop(out, token)
        need_wait = True
        state["restarts"] += 1
    reason = "training_budget_exhausted" if state["remaining_wall_seconds"] < 1 else "restart_limit"
    publish(out, session, state, "finished", reason=reason, returncode=0)
    return 0
