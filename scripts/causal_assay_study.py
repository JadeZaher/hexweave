"""Pin the six-arm account and reject standalone assay ownership."""

import hashlib
import os
from pathlib import Path

from scripts import causal_assay as assay
from scripts.causal_observer import contained, encoded, load
from scripts.pulse_io import digest


ARMS = [{"seed": seed, "rule": rule} for seed in (7, 17, 29) for rule in ("dense", "magnitude")]
BUDGET = {"maximum_opportunities": 1601, "maximum_wall_millis": 1_100_000,
          "maximum_attempts": 1, "disk_bytes_per_attempt": 64 * 1024**2}
LIMITS = {"opportunities": 1600, "wall_millis": 1_100_000, "stop_reserve_millis": 15_000,
          "checkpoint_every": 64, "evaluation_targets": 1920, "generation_cases": 48, "generation_bytes": 8}
WHOLE_MILLIS = 7_200_000
DISK_CAP = 512 * 1024**2


def operation(value: dict) -> None:
    from scripts.causal_contract import integer, sha
    if (set(value) != {"kind", "study", "arm"} or value["kind"] != "retained_assay"
            or not integer(value["arm"], 0, 5) or not isinstance(value["study"], dict)
            or set(value["study"]) != {"path", "sha256"} or not sha(value["study"]["sha256"])
            or not isinstance(value["study"]["path"], str)):
        raise ValueError("closed study identity and arm ordinal required")


def declaration(root: Path, spec: dict) -> tuple[Path, dict]:
    from scripts.causal_contract import pin
    path = pin(root, spec, 16_384)
    value = load(path, 16_384)
    fields = {"schema", "protocol", "run_id", "common_identity", "arms", "budget", "limits", "whole_wall_millis", "disk_cap_bytes"}
    if (set(value) != fields or type(value["schema"]) is not int or value["schema"] != 1
            or value["protocol"] != assay.PROTOCOL or value["arms"] != ARMS
            or value["budget"] != BUDGET or value["limits"] != LIMITS
            or value["whole_wall_millis"] != WHOLE_MILLIS or value["disk_cap_bytes"] != DISK_CAP
            or not isinstance(value["run_id"], str) or len(value["run_id"]) != 32
            or any(c not in "0123456789abcdef" for c in value["run_id"])):
        raise ValueError("invalid conserved study declaration")
    expected = root / "work/causal-studies" / value["run_id"] / "study.json"
    if path != expected.resolve(strict=True):
        raise ValueError("noncanonical study declaration path")
    return path.parent, value


def request(root: Path, value: dict) -> None:
    operation(value["operation"])
    _, study = declaration(root, value["operation"]["study"])
    arm = ARMS[value["operation"]["arm"]]
    common = {key: val for key, val in value["identity"].items() if key != "config_sha256"}
    if (value["config"]["seed"] != arm["seed"] or value["config"]["profile"] != "retained-" + arm["rule"]
            or common != study["common_identity"] or value["resume"] is not None
            or value["budget"] != BUDGET or value["limits"] != LIMITS):
        raise ValueError("assay request escaped its fixed fresh arm/account")


def owner_context(root: Path, value: dict) -> None:
    request(root, value)
    operation_value = value["operation"]
    directory, _ = declaration(root, operation_value["study"])
    marker = load(directory / f"arm-{operation_value['arm']}-entered.json", 4096)
    driver = {"pid": int(os.environ["HEXWEAVE_CAUSAL_DRIVER_PID"]),
              "creation_filetime": int(os.environ["HEXWEAVE_CAUSAL_DRIVER_CREATED"])}
    expected = {"schema": 1, "study_sha256": operation_value["study"]["sha256"],
                "arm": operation_value["arm"], "driver": driver,
                "request_sha256": hashlib.sha256(encoded(value)).hexdigest()}
    if (marker != expected or os.environ.get("HEXWEAVE_CAUSAL_STUDY_SHA256") != expected["study_sha256"]
            or os.environ.get("HEXWEAVE_CAUSAL_STUDY_ARM") != str(expected["arm"])):
        raise ValueError("assay owner lacks live conserved study handoff")
