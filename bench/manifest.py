"""Freeze benchmark identity and bounded inputs; see bench/AGENTS.md."""

import math
import re
from pathlib import Path
from urllib.parse import urlsplit

from .files import (MAX_DATASET_BYTES, MAX_MANIFEST_BYTES, decode_json,
                    read_bounded, relative_file, sha256_bytes, sha256_file)


SCHEMA = "hexweave-local-benchmark-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f"{label} requires exactly: {keys}")


def text_value(value, label, maximum=256):
    if (not isinstance(value, str) or not value or len(value) > maximum
            or any(ord(character) < 32 for character in value)):
        raise ValueError(f"Invalid {label}")
    if any(marker in value.lower() for marker in ("replace_", "unresolved", "placeholder")):
        raise ValueError(f"Unpinned {label}")
    return value


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")


def number(value, low, high, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")


def digest(value, label):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


def endpoint(value):
    """Accept direct numeric loopback endpoints without URL authority extras."""
    text_value(value, "endpoint")
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1")
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("Endpoint must be http://127.0.0.1:PORT or http://[::1]:PORT")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Invalid endpoint port") from error
    integer(port, 1, 65535, "endpoint port")
    return parsed.hostname, port


def _model(value, base):
    exact_keys(value, "id revision kind artifacts", "model")
    text_value(value["id"], "model ID")
    if not isinstance(value["revision"], str) or not REVISION.fullmatch(value["revision"]):
        raise ValueError("Model revision must be an immutable 40- or 64-character hex digest")
    if value["kind"] not in ("base", "instruct"):
        raise ValueError("Model kind must be base or instruct")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or not 3 <= len(artifacts) <= 128:
        raise ValueError("Pin 3 to 128 model/tokenizer/template artifacts")
    roles, paths = set(), set()
    for artifact in artifacts:
        exact_keys(artifact, "role path sha256", "artifact")
        if artifact["role"] not in ("weights", "tokenizer", "template"):
            raise ValueError("Invalid artifact role")
        roles.add(artifact["role"])
        digest(artifact["sha256"], "artifact hash")
        path = relative_file(base, artifact["path"])
        if path in paths:
            raise ValueError("Duplicate artifact path")
        paths.add(path)
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {artifact['role']}")
    if roles != {"weights", "tokenizer", "template"}:
        raise ValueError("Weights, tokenizer and template artifact hashes are required")


def _backend(value):
    exact_keys(value, "name version endpoint model_alias compute_dtype quantization context_tokens threads kv_cache_dtype", "backend")
    for field in ("name", "version", "model_alias", "compute_dtype", "quantization", "kv_cache_dtype"):
        text_value(value[field], f"backend {field}")
    if value["version"].lower() in ("latest", "main", "master", "head", "unknown"):
        raise ValueError("Backend version must be pinned")
    if value["kv_cache_dtype"].lower() in ("auto", "default", "unknown"):
        raise ValueError("KV cache dtype must be declared explicitly")
    endpoint(value["endpoint"])
    integer(value["context_tokens"], 128, 131072, "context_tokens")
    integer(value["threads"], 1, 4, "backend threads")


def _generation(value):
    exact_keys(value, "max_tokens temperature top_p seed stop", "generation")
    integer(value["max_tokens"], 1, 512, "max_tokens")
    number(value["temperature"], 0, 2, "temperature")
    number(value["top_p"], 0.01, 1, "top_p")
    integer(value["seed"], 0, 2**32 - 1, "seed")
    if not isinstance(value["stop"], list) or len(value["stop"]) > 8:
        raise ValueError("Stop must contain at most eight strings")
    for stop in value["stop"]:
        if not isinstance(stop, str) or not 1 <= len(stop.encode("utf-8")) <= 128:
            raise ValueError("Invalid stop string")
    if len(set(value["stop"])) != len(value["stop"]):
        raise ValueError("Duplicate stop strings")


def _budget(value):
    exact_keys(value, "examples repetitions warmups timeout_seconds max_wall_seconds max_response_bytes", "budget")
    for key, low, high in (("examples", 1, 128), ("repetitions", 1, 10), ("warmups", 0, 3),
                           ("timeout_seconds", 1, 120), ("max_wall_seconds", 1, 600),
                           ("max_response_bytes", 1024, 1048576)):
        integer(value[key], low, high, key)
    if value["timeout_seconds"] > value["max_wall_seconds"]:
        raise ValueError("Request timeout cannot exceed total wall-time budget")


def _dataset(value, base, count):
    exact_keys(value, "id revision split path sha256 prompt_format", "dataset")
    for field in ("id", "revision", "split", "prompt_format"):
        text_value(value[field], f"dataset {field}")
    digest(value["sha256"], "dataset hash")
    content = read_bounded(relative_file(base, value["path"]), MAX_DATASET_BYTES)
    if sha256_bytes(content) != value["sha256"]:
        raise ValueError("Dataset hash mismatch")
    rows, identifiers = [], set()
    for line in content.splitlines():
        row = decode_json(line)
        exact_keys(row, "id prompt expected", "dataset row")
        text_value(row["id"], "example ID", 128)
        if row["id"] in identifiers:
            raise ValueError("Duplicate dataset example ID")
        identifiers.add(row["id"])
        if not isinstance(row["prompt"], str) or not 1 <= len(row["prompt"].encode("utf-8")) <= 16384:
            raise ValueError("Prompt must contain 1 to 16384 UTF-8 bytes")
        if row["expected"] is not None and (not isinstance(row["expected"], str)
                                           or len(row["expected"].encode("utf-8")) > 16384):
            raise ValueError("Expected output must be null or a bounded string")
        rows.append(row)
        if len(rows) > 1000:
            raise ValueError("Dataset exceeds 1000 examples")
    if len(rows) < count:
        raise ValueError("Dataset has fewer rows than budget.examples")
    return rows[:count]


def load_manifest(path):
    """Verify all local inputs before returning an immutable input snapshot."""
    path = Path(path).resolve()
    content = read_bounded(path, MAX_MANIFEST_BYTES)
    value = decode_json(content)
    exact_keys(value, "schema purpose model backend dataset generation budget", "manifest")
    if value["schema"] != SCHEMA:
        raise ValueError("Unknown benchmark schema")
    if value["purpose"] not in ("deployment_comparison", "software_fixture"):
        raise ValueError("Invalid benchmark purpose")
    _backend(value["backend"])
    _generation(value["generation"])
    if value["generation"]["max_tokens"] >= value["backend"]["context_tokens"]:
        raise ValueError("Generation must leave context space for the prompt")
    _budget(value["budget"])
    _model(value["model"], path.parent)
    rows = _dataset(value["dataset"], path.parent, value["budget"]["examples"])
    return {"manifest": value, "manifest_sha256": sha256_bytes(content), "rows": rows}
