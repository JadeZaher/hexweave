"""Derive a pinned plain-text training corpus from a TinyStories-format file (stories split by `<|endoftext|>`)."""

import argparse
import hashlib
import json
from pathlib import Path

SEPARATOR = b"<|endoftext|>"
END = b"\x03"


def stories(raw: bytes) -> list[bytes]:
    """Complete stories: the leading fragment before the first separator is dropped."""
    parts = raw.split(SEPARATOR)
    return [story.strip() for story in parts[1:] if story.strip()]


def split(all_stories: list[bytes], every: int) -> tuple[bytes, bytes]:
    """Every `every`-th story is held out; each story ends with one END byte."""
    train = b"".join(story + END for index, story in enumerate(all_stories) if index % every)
    heldout = b"".join(story + END for index, story in enumerate(all_stories) if not index % every)
    return train, heldout


def prepare(source: Path, out: Path, every: int) -> dict:
    raw = source.read_bytes()
    train, heldout = split(stories(raw), every)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data in (("train.txt", train), ("heldout.txt", heldout)):
        (out / name).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "schema": 1, "kind": "byte-corpus", "end_byte": END[0], "heldout_every": every,
        "source": {"path": str(source), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heldout-every", type=int, default=20)
    arguments = parser.parse_args()
    manifest = prepare(arguments.source.resolve(strict=True), arguments.out, arguments.heldout_every)
    print(json.dumps(manifest["files"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
