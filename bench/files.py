"""Validate bounded input and owned output files; see bench/AGENTS.md."""

import hashlib
import json
import math
from pathlib import Path


MAX_MANIFEST_BYTES = 128 * 1024
MAX_DATASET_BYTES = 2 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number exceeds finite range")
    return parsed


def decode_json(raw):
    """Decode JSON while rejecting duplicate keys and nonfinite numbers."""
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="strict")
        return json.loads(raw, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant, parse_float=_finite_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid UTF-8 JSON") from error


def read_bounded(path, maximum):
    """Read at most one declared input limit."""
    with Path(path).open("rb") as handle:
        content = handle.read(maximum + 1)
    if len(content) > maximum:
        raise ValueError(f"Input exceeds {maximum} byte limit")
    return content


def sha256_bytes(content):
    """Hash the exact bytes used by an operation."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(path):
    """Stream an artifact hash without allocating its complete payload."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file(base, value):
    """Resolve a manifest-relative path without expanding shell expressions."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Expected a nonempty local file path")
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise ValueError(f"Required local file does not exist: {value}")
    return path.resolve()


def _no_links(path, root):
    cursor = path
    while cursor != root:
        if cursor.is_symlink() or cursor.is_junction():
            raise ValueError("Output path cannot contain symlinks or junctions")
        cursor = cursor.parent


def exclusive_output(root, value):
    """Create a new result below this repository's work directory."""
    root = Path(root).resolve()
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    lexical = Path(candidate.absolute())
    if ".." in lexical.parts or not lexical.is_relative_to(root / "work"):
        raise ValueError("Result path must be inside this repository's work/")
    if lexical == root / "work":
        raise ValueError("Result path must name a file under work/")
    _no_links(lexical, root)
    resolved = lexical.resolve()
    if not resolved.is_relative_to(root / "work"):
        raise ValueError("Result path escaped work/")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _no_links(lexical, root)
    return resolved.open("x", encoding="utf-8", newline="\n")


def write_record(handle, record):
    """Flush one finite JSON record to an exclusively owned result file."""
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
