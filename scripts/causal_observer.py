"""Bounded synced events and terminal publication; see causal_OWNERSHIP.md."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.pulse_io import finite_tree, unique_object


JSON_CAP = 131_072
EVENT_CAP = 8 * 1024**2


def encoded(value: dict) -> bytes:
    if not finite_tree(value):
        raise ValueError("nonfinite causal observer value")
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(path: Path, cap: int = JSON_CAP) -> dict:
    with path.open("rb") as stream:
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError("causal JSON exceeds bounded read")
    value = json.loads(data, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or not finite_tree(value):
        raise ValueError("causal JSON must be a finite object")
    return value


def write(path: Path, value: dict, *, immutable: bool = False) -> None:
    data = encoded(value)
    if len(data) > JSON_CAP:
        raise ValueError("causal observer record exceeds bound")
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if immutable:
        os.link(pending, path)
        pending.unlink()
    else:
        pending.replace(path)


class Observer:
    def __init__(self, directory: Path, sequence: int):
        self.directory, self.sequence = directory, sequence - 1
        self.previous = "0" * 64
        self.bytes = 0

    def emit(self, kind: str, **fields) -> dict:
        value = {"schema": 1, "sequence": self.sequence + 1, "previous_sha256": self.previous,
                 "utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
        data = encoded(value) + b"\n"
        if len(data) > 16_384 or self.bytes + len(data) > EVENT_CAP:
            raise ValueError("causal observer event bound exceeded")
        with (self.directory / "events.jsonl").open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1
        self.previous = hashlib.sha256(data).hexdigest()
        self.bytes += len(data)
        acknowledgement = {"schema": 1, "sequence": self.sequence, "event_sha256": self.previous}
        write(self.directory / "observer.json", acknowledgement)
        return acknowledgement


def request_stop(directory: Path, reason: str, observer: Observer) -> None:
    path = directory / "worker-stop.json"
    if not path.exists():
        acknowledgement = observer.emit("cooperative_stop_requested", reason=reason)
        write(path, {"schema": 1, "reason": reason, "observer_sequence": acknowledgement["sequence"]}, immutable=True)


def contained(root: Path, relative: str, *, exists: bool = True) -> Path:
    path = root / relative
    if path.is_symlink():
        raise ValueError("causal path cannot be a symlink")
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("causal path escapes workspace")
    return resolved
