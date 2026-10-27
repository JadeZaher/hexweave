"""Closed retained-text request and source identity; see the assay contract."""

import hashlib
import json
from pathlib import Path

from scripts.causal_observer import contained
from scripts.pulse_io import digest


PROTOCOL = "causal-retained-text-assay-v1"
CONTRACT = "docs/CAUSAL-RETAINED-TEXT-ASSAY-V1-CONTRACT.md"
CONTRACT_SHA = "e49b0bc82f13501c5ebf628d91b7383b58a19792af01d91081ac2aa9f676619d"
MANIFEST_SHA = "0a77b8fe038c8139314e7331417ced5aa853805ec26562680c3d770a01ade6db"
ROWS_SHA = "0ad6de82f45fb0cff217ad34ac2ebe60eaff2ef1ce32399ef892b36ecbfc40b9"
SCHEDULE_SHA = "7ceefd7f72531f01c3336af6779d89e4e1ac5cfd649ba2b5f94f62004501ad3c"
PASSES = [0, 1, 2, 4, 8, 16, 17, 20]
GENERATIONS = [0, 4, 8, 16, 17, 20]


def identity() -> dict:
    return {"schema": "retained-text-source-v1", "manifest_sha256": MANIFEST_SHA,
            "artifact_sha256": ROWS_SHA, "source_pin_manifest_sha256": SCHEDULE_SHA,
            "dataset_version": "retained-text-fixed64-v1", "source_as_of": "synthetic",
            "split": "train", "rows": 32, "completion_targets_per_epoch": 80}


def configuration(value: dict) -> dict:
    if value.get("profile") not in ("retained-dense", "retained-magnitude") or value.get("seed") not in (7, 17, 29):
        raise ValueError("unsupported assay arm or seed")
    return {"protocol": PROTOCOL, "profile": value["profile"], "anchor_interval": 1,
            "replay_selection": "none-anchor-only-v1"}


def limits(value: dict) -> None:
    from scripts.causal_contract import integer
    fixed = {"evaluation_targets": 1920, "generation_cases": 48, "generation_bytes": 8}
    bounds = {"opportunities": (1, 1600), "wall_millis": (5001, 6_600_000),
              "stop_reserve_millis": (5000, 60_000), "checkpoint_every": (1, 64)}
    if (not isinstance(value, dict) or set(value) != set(fixed) | set(bounds)
            or any(type(value[key]) is not int or value[key] != expected for key, expected in fixed.items())
            or any(not integer(value[key], *span) for key, span in bounds.items())
            or value["wall_millis"] <= value["stop_reserve_millis"]):
        raise ValueError("invalid finite retained-text limits")


def source(root: Path) -> list[dict]:
    for name, sha, cap in (("manifest.json", MANIFEST_SHA, 16_384),
                           ("rows.jsonl", ROWS_SHA, 65_536), ("schedule.json", SCHEDULE_SHA, 32_768)):
        path = contained(root, name)
        if path.stat().st_size > cap or digest(path) != sha:
            raise ValueError("retained-text source pin changed")
    raw = (root / "rows.jsonl").read_bytes()
    if len(raw) > 65_536 or hashlib.sha256(raw).hexdigest() != ROWS_SHA:
        raise ValueError("assay rows changed during open")
    return [json.loads(line) for line in raw.splitlines()]


def request(root: Path, value: dict, files: dict) -> None:
    from scripts.causal_assay_study import request as study_request
    study_request(root, value)
    if (value["identity"]["data"] != identity()
            or value["budget"]["maximum_opportunities"] != 1601
            or value["budget"]["disk_bytes_per_attempt"] != 64 * 1024**2
            or files.get(CONTRACT) != CONTRACT_SHA):
        raise ValueError("assay protocol/source/budget mismatch")
    source(contained(root, value["data_root"]))
    if value["resume"] is None and value["limits"]["opportunities"] != 1600:
        raise ValueError("fresh assay must declare full fixed course")


def cursor(n: int) -> dict:
    from scripts.causal_contract import integer
    if not integer(n, 0, 1600):
        raise ValueError("invalid assay physical exposure")
    passed, position = n // 80, n % 80 // 5
    row = ((position if passed % 2 == 0 else 15 - position) + passed % 16) % 16
    group = "foundation" if passed < 16 else "retention-stress"
    return {"schema": 2, "data": identity(), "row_index": 0 if n == 1600 else row,
            "example_id": "complete" if n == 1600 else f"{group}-{row:02}",
            "completion_offset": n % 5, "epoch": passed, "supervised_targets": n,
            "answer_targets": n // 5 * 4 + min(n % 5, 4), "citation_targets": 0, "format_targets": n // 5}
