"""External request pins and finite budgets; see causal_OWNERSHIP.md."""

import hashlib
from pathlib import Path

from scripts.causal_observer import contained, encoded, load
from scripts.pulse_io import digest
from scripts import causal_prompts, causal_assay


PROTOCOL = "causal-prefix-v3"
PROTOCOL_SHA256 = "f54b9dc8da314cf41e95a4cd40362e8fcad39a547a6ad442d31c2c676edac3a1"
V2_PROTOCOL_SHA256 = "b6d725b398867675701b9f9bf23484e1f2e72547f3029086d38561859c10fab3"
BASE_PROTOCOL_SHA256 = "762b6ae6abacae6e047ed5fc2db50cb95471475878ee2a20b8f8f2629fbcd478"
HASH_CAP = 128 * 1024**2


def integer(value, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def sha(value) -> bool:
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def configuration(value: dict) -> None:
    expected = {"schema": 1, "protocol": PROTOCOL, "profile": "candidate", "threads": 1,
                "anchor_interval": 4, "max_visits": 4, "history_capacity": 8, "context_bytes": 1024,
                "evidence_rule": "threshold-one-canonical-zero-v1", "route_credit": "detached-u-experimental",
                "predictor_rule": "joint-adjacent-raw-row-mean-v1", "anchor_batch": "current-anchor-forward-v1",
                "replay_selection": "newest4-compatible-v1"}
    if isinstance(value, dict) and value.get("protocol") == causal_assay.PROTOCOL:
        expected.update(causal_assay.configuration(value))
    if (not isinstance(value, dict) or set(value) != set(expected) | {"seed"}
            or not integer(value.get("seed"), 0, 2**64 - 1)
            or any(type(value[key]) is not type(wanted) or value[key] != wanted for key, wanted in expected.items())):
        raise ValueError("unsupported owned causal configuration or joint rule")


def pin(root: Path, spec: dict, cap: int) -> Path:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"} or not sha(spec["sha256"]):
        raise ValueError("external file identity required")
    path = contained(root, spec["path"])
    if not path.is_file() or path.stat().st_size > cap or digest(path) != spec["sha256"]:
        raise ValueError("pinned causal file changed or exceeds bound")
    return path


def budget(value: dict) -> None:
    bounds = {"maximum_opportunities": (1, 64_000_000), "maximum_wall_millis": (5001, 86_400_000),
              "maximum_attempts": (1, 64), "disk_bytes_per_attempt": (1_048_576, 2 * 1024**3)}
    if not isinstance(value, dict) or set(value) != set(bounds) or any(not integer(value[key], *span) for key, span in bounds.items()):
        raise ValueError("finite explicit cumulative budget required")


def limits(value: dict, prompt_only: bool = False, assay: bool = False) -> None:
    if assay:
        causal_assay.limits(value)
        return
    bounds = {"opportunities": (0, 0) if prompt_only else (1, 1_000_000), "wall_millis": (5001, 7_200_000),
              "stop_reserve_millis": (5000, 60_000), "checkpoint_every": (0, 0) if prompt_only else (1, 1_000_000),
              "evaluation_targets": (0, 0) if prompt_only else (1, 65_536), "generation_cases": (1, 40 if prompt_only else 32), "generation_bytes": (1, 192)}
    if not isinstance(value, dict) or set(value) != set(bounds) or any(not integer(value[key], *span) for key, span in bounds.items()):
        raise ValueError("finite explicit attempt limits required")
    if value["wall_millis"] <= value["stop_reserve_millis"] or value["checkpoint_every"] > value["opportunities"]:
        raise ValueError("invalid stop reserve or checkpoint cadence")


def source_files(root: Path, spec: dict) -> dict:
    manifest = load(pin(root, spec, 131_072))
    files = manifest.get("files")
    if set(manifest) != {"schema", "files"} or type(manifest.get("schema")) is not int or manifest["schema"] != 1 or not isinstance(files, dict) or not 1 <= len(files) <= 1024:
        raise ValueError("bounded source manifest required")
    required = {"Cargo.toml", "Cargo.lock", ".cargo/config.toml", "docs/CAUSAL-PREFIX-CONTRACT.md",
                "docs/CAUSAL-PREFIX-V2-CONTRACT.md", "docs/CAUSAL-PREFIX-V3-CONTRACT.md",
                "datasets/tiny/train.txt", "datasets/tiny/development-t2.txt"}
    for directory, pattern in (("src", "*.rs"), ("scripts", "*.py"), ("tests", "*.rs"), ("tests", "*.py")):
        required.update(path.relative_to(root).as_posix() for path in (root / directory).rglob(pattern))
    if not required <= files.keys():
        raise ValueError("source manifest omits required code/build inputs")
    if (files["docs/CAUSAL-PREFIX-CONTRACT.md"] != BASE_PROTOCOL_SHA256 or
            files["docs/CAUSAL-PREFIX-V2-CONTRACT.md"] != V2_PROTOCOL_SHA256 or
            files["docs/CAUSAL-PREFIX-V3-CONTRACT.md"] != PROTOCOL_SHA256):
        raise ValueError("durable protocol documents differ from compiled contract")
    total = 0
    for name, expected in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or name.startswith(("work/", ".git/")):
            raise ValueError("source manifest path is not a source input")
        path = pin(root, {"path": name, "sha256": expected}, 1_048_576)
        total += path.stat().st_size
        if total > 32 * 1024**2:
            raise ValueError("source manifest total exceeds bound")
    return files


def validate(root: Path, request: dict) -> dict:
    keys = {"schema", "protocol", "binary", "source_manifest", "build_receipt", "identity", "config", "data_root", "budget", "limits", "resume", "operation"}
    if set(request) != keys or type(request["schema"]) is not int or request["schema"] != 1 or request["protocol"] not in (PROTOCOL, causal_assay.PROTOCOL):
        raise ValueError("unsupported causal owner request schema")
    budget(request["budget"])
    prompt_only = causal_prompts.operation(request["operation"])
    assay = request["protocol"] == causal_assay.PROTOCOL
    if assay != (request["operation"].get("kind") == "retained_assay"):
        raise ValueError("mixed assay protocol/operation")
    limits(request["limits"], prompt_only, assay)
    prompt_path = None
    if prompt_only:
        if request["resume"] is None:
            raise ValueError("prompt evaluation requires completed parent")
        prompt_path, _ = causal_prompts.read(root, request["operation"]["prompts"],
            request["limits"]["generation_cases"], request["limits"]["generation_bytes"])
    identity, config = request["identity"], request["config"]
    if not isinstance(identity, dict) or not isinstance(config, dict):
        raise ValueError("immutable identity/config objects required")
    if identity.get("protocol") != request["protocol"] or identity.get("protocol_sha256") != (causal_assay.CONTRACT_SHA if assay else PROTOCOL_SHA256) or identity.get("kernel") not in ("scalar", "avx2"):
        raise ValueError("causal protocol/resolved kernel must be pinned")
    configuration(config)
    if config["protocol"] != request["protocol"]:
        raise ValueError("mixed config protocol")
    binary = pin(root, request["binary"], HASH_CAP)
    sources = source_files(root, request["source_manifest"])
    if assay:
        causal_assay.request(root, request, sources)
    if (identity.get("runtime_sha256") != request["binary"]["sha256"] or
            identity.get("source_sha256") != request["source_manifest"]["sha256"] or
            identity.get("config_sha256") != hashlib.sha256(encoded(config)).hexdigest()):
        raise ValueError("immutable source/runtime/config identity mismatch")
    build = load(pin(root, request["build_receipt"], 65_536), 65_536)
    expected = {"protocol": request["protocol"], "status": "completed", "binary_sha256": request["binary"]["sha256"],
                "source_manifest_sha256": request["source_manifest"]["sha256"]}
    if any(build.get(key) != value for key, value in expected.items()):
        raise ValueError("separate build receipt does not bind source and binary")
    data = contained(root, request["data_root"])
    pin(root, {"path": str(data / "manifest.json"), "sha256": identity["data"]["manifest_sha256"]}, 65_536)
    return {"binary": binary, "data_root": data, "source_files": sources,
            "source_manifest": contained(root, request["source_manifest"]["path"]), "prompts": prompt_path}
