"""Bounded local experiment receipts; see scripts/AGENTS.md."""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


LOG_LIMIT = 1_048_576


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            value.update(block)
    return value.hexdigest()


def event(directory: Path, kind: str, **fields) -> None:
    value = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish(directory: Path, report: dict) -> None:
    pending = directory / "report.pending.json"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(directory / "report.json")


def finite_tree(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def unique_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate result key: {key}")
        result[key] = value
    return result


def check_metrics(result: dict, steps: int, anchor_interval: int) -> None:
    for key in ("initial_loss", "final_loss", "seconds"):
        value = result.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"missing or invalid result metric: {key}")
    if type(result.get("parameters")) is not int or result["parameters"] <= 0:
        raise ValueError("missing or invalid parameter count")
    if result.get("all_learned_weights_ternary") is not True or result.get("fp32_master_weights") is not False:
        raise ValueError("unexpected precision contract")
    anchors = (steps - 1) // anchor_interval + 1
    counts = result.get("counts")
    if not isinstance(counts, dict) or type(counts.get("anchor_steps")) is not int:
        raise ValueError("missing anchor accounting")
    if counts["anchor_steps"] != anchors or type(result.get("training_target_exposures")) is not int:
        raise ValueError("invalid anchor accounting")
    if result["training_target_exposures"] != anchors:
        raise ValueError("task exposure does not match anchors")
    state = result.get("state_bytes")
    names = ("packed_weight_payload", "shared_integer_evidence",
             "resident_integer_evidence_current", "resident_integer_evidence_peak",
             "resident_integer_evidence_bound")
    if not isinstance(state, dict) or any(type(state.get(key)) is not int or state[key] < 0 for key in names):
        raise ValueError("missing or invalid state accounting")
    if not state["resident_integer_evidence_current"] <= state["resident_integer_evidence_peak"] <= state["resident_integer_evidence_bound"]:
        raise ValueError("resident evidence exceeds its declared bound")


def read_result(path: Path, steps: int, seed: int, mode: str,
                residents: int, anchor_interval: int) -> dict:
    if path.stat().st_size > LOG_LIMIT:
        raise ValueError("child output exceeds the receipt limit")
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(result, dict) or not finite_tree(result):
        raise ValueError("child result must be a finite JSON object")
    expected = {"protocol": "ternary-pulse-v1", "completed_steps": steps,
                "seed": seed, "mode": mode, "resident_limit": residents,
                "anchor_interval": anchor_interval}
    for key, value in expected.items():
        if type(result.get(key)) is not type(value) or result.get(key) != value:
            raise ValueError(f"child result mismatch for {key}")
    check_metrics(result, steps, anchor_interval)
    return result


def owned_directory(root: Path) -> Path:
    root = root.resolve(strict=True)
    for relative in ("work", "work/pulse"):
        path = root / relative
        if path.resolve().is_relative_to(root) is False or path.is_symlink():
            raise ValueError("runtime directory must stay inside the repository")
        path.mkdir(exist_ok=True)
    return root / "work/pulse"
