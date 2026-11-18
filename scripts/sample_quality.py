"""Score read-only checkpoint samples: known words, distinct 4-grams, word length and loops.

Usage: python -B scripts/sample_quality.py --config <run.json> --checkpoint <aNNNNNNNNNNNN>
       --binary <hexweave.exe> --out <result.json outside the run directory>
Metric definitions follow work/inference/reference-decoding-20261111/record.py; see scripts/AGENTS.md.
"""

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ["Once upon a time", "One day, a little girl named Lily", "Tom and his dog went to the", "The sun was"]
NUCLEUS = "nucleus T0.8 p0.9"
SETTINGS = [("greedy", "greedy", [])] + [
    (f"{NUCLEUS} seed {seed}", NUCLEUS, ["--top-p", "0.9", "--temperature", "0.8", "--seed", str(seed)])
    for seed in (1, 2, 3)
]
LIMIT = 200
WORD = re.compile(r"[a-z]+")


def vocabulary(path: Path) -> set[str]:
    """Lower-case words seen at least three times in the public train split, read line by line."""
    counts = collections.Counter()
    with path.open("rb") as train:
        for line in train:
            counts.update(WORD.findall(line.decode("utf-8", "replace").lower()))
    return {word for word, count in counts.items() if count >= 3}


def loops(data: bytes) -> bool:
    """A 64-byte tail with a period of at most 32 bytes (the reference definition)."""
    tail = data[-64:]
    return len(tail) == 64 and any(all(tail[i] == tail[i - p] for i in range(p, 64)) for p in range(1, 33))


def metrics(data: bytes, words: set[str]) -> dict:
    """Loop flag, distinct byte 4-gram share, known-word share and mean word length of one sample."""
    grams = [data[i:i + 4] for i in range(len(data) - 3)]
    tokens = WORD.findall(data.decode("utf-8", "replace").lower())
    return {"loops": loops(data), "distinct_4grams": len(set(grams)) / max(1, len(grams)),
            "known_words": sum(t in words for t in tokens) / max(1, len(tokens)),
            "mean_word_length": sum(map(len, tokens)) / max(1, len(tokens)), "words": len(tokens)}


def summarize(rows: list[dict]) -> dict:
    """Per-sample means of each metric and the share of looping samples."""
    count = len(rows)
    mean = lambda key: sum(row[key] for row in rows) / count
    return {"samples": count, "known_words": mean("known_words"), "distinct_4grams": mean("distinct_4grams"),
            "mean_word_length": mean("mean_word_length"), "loop_share": mean("loops")}


def fingerprint(report: dict, connected: bool) -> str:
    """The unchanged state fingerprint; connected reports must also show an unchanged checkpoint."""
    pairs = [("before_sha256", "after_sha256")]
    if connected:
        pairs.append(("checkpoint_before_sha256", "checkpoint_after_sha256"))
    for before, after in pairs:
        if not report.get(before) or report.get(before) != report.get(after):
            raise SystemExit(f"completion report lacks or changed {before}/{after}")
    return report["after_sha256"]


def resolve(path: str) -> Path:
    """A config path, relative to the repository root unless absolute."""
    return (ROOT / path).resolve()


def complete(binary: Path, config: Path, checkpoint: str, prompt: str, sampling: list[str]) -> dict:
    """Run the binary's read-only `complete` (cwd: the repository root) and return its JSON report."""
    command = [str(binary), "complete", "--config", str(config), "--checkpoint", checkpoint,
               "--prompt", prompt, "--max-new-bytes", str(LIMIT), *sampling]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode("utf-8", "replace"))
    return json.loads(result.stdout)


def parse_args() -> argparse.Namespace:
    """Command-line arguments; file paths become absolute against the caller's directory, because
    the binary runs with the repository root as its working directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.config, args.binary, args.out = (Path(path).resolve() for path in (args.config, args.binary, args.out))
    return args


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["data"]["kind"] != "corpus":
        raise SystemExit("sample quality needs a corpus run")
    if args.out.is_relative_to(resolve(config["out"])):
        raise SystemExit("--out must be outside the run directory")
    words = vocabulary(resolve(config["data"]["root"]) / "train.txt")
    connected, rows = "connected" in config, []
    for prompt in PROMPTS:
        for name, group, sampling in SETTINGS:
            report = complete(args.binary, args.config, args.checkpoint, prompt, sampling)
            rows.append({"prompt": prompt, "setting": name, "group": group, "anchors": report["anchors"],
                         "stop_reason": report["stop_reason"], "fingerprint": fingerprint(report, connected),
                         "text": report["text"], **metrics(bytes(report["raw_bytes"]), words)})
    if len({row["fingerprint"] for row in rows}) != 1:
        raise SystemExit("completions loaded different states from one checkpoint")
    groups = {group: summarize([row for row in rows if row["group"] == group]) for group in ("greedy", NUCLEUS)}
    groups["overall"] = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": str(args.config), "checkpoint": args.checkpoint,
                                    "binary": str(args.binary), "max_new_bytes": LIMIT, "summary": groups,
                                    "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'setting':<20}{'n':>4}{'known':>8}{'4gram':>8}{'wordlen':>9}{'loops':>7}")
    for name, value in groups.items():
        print(f"{name:<20}{value['samples']:>4}{value['known_words']:>8.3f}{value['distinct_4grams']:>8.3f}"
              f"{value['mean_word_length']:>9.2f}{value['loop_share']:>7.2f}")


if __name__ == "__main__":
    main()
