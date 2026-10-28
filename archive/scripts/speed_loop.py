"""Measure repeated fixed-protocol fixtures; see docs/PERFORMANCE-LOOPS.md."""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def deterministic(result: dict) -> dict:
    result = copy.deepcopy(result)
    result.pop("seconds", None)
    result.get("arithmetic", {}).pop("execution", None)
    return result


def measure(binary: Path, kernel: str, repetitions: int) -> dict:
    if not 3 <= repetitions <= 15:
        raise ValueError("repetitions must be 3..15")
    if kernel not in ("auto", "scalar", "avx2"):
        raise ValueError("unsupported kernel")
    binary = binary.resolve(strict=True)
    binary_hash = digest(binary)
    environment = dict(os.environ, HEXWEAVE_KERNEL=kernel)
    cases = []
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    for mode in ("pulse", "anchor-only"):
        for seed in (7, 8, 9):
            samples, walls, reference = [], [], None
            for repetition in range(repetitions + 1):
                started = time.perf_counter()
                completed = subprocess.run(
                    [str(binary), "pulse", "--steps", "256", "--seed", str(seed),
                     "--mode", mode, "--residents", "1", "--anchor-interval", "4"],
                    capture_output=True, timeout=10, check=True, env=environment,
                    stdin=subprocess.DEVNULL, creationflags=flags,
                )
                wall = time.perf_counter() - started
                if len(completed.stdout) > 1_048_576:
                    raise ValueError("fixture output unexpectedly large")
                result = json.loads(completed.stdout)
                seconds = result.get("seconds")
                if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("fixture timing must be finite and positive")
                identity = deterministic(result)
                if reference is not None and reference != identity:
                    raise ValueError(f"non-repeatable result: {mode} seed {seed}")
                reference = identity
                if repetition:
                    samples.append(result["seconds"])
                    walls.append(wall)
            median = statistics.median(samples)
            cases.append({"mode": mode, "seed": seed, "samples_seconds": samples,
                          "median_seconds": median,
                          "mad_seconds": statistics.median(abs(x - median) for x in samples),
                          "median_process_wall_seconds": statistics.median(walls),
                          "result": reference})
    if digest(binary) != binary_hash:
        raise ValueError("binary changed during the measurement")
    return {"binary": str(binary), "sha256": binary_hash, "requested_kernel": kernel,
            "repetitions": repetitions, "warmups_per_case": 1, "cases": cases,
            "interpretation": "same-fixture release timing, not time-to-quality or a large model forecast"}


def compare(current: dict, baseline: dict) -> list[dict]:
    if len(current["cases"]) != len(baseline["cases"]):
        raise ValueError("case count changed")
    comparisons = []
    for new, old in zip(current["cases"], baseline["cases"]):
        if deterministic(new["result"]) != deterministic(old["result"]):
            raise ValueError(f"protocol changed: {new['mode']} seed {new['seed']}")
        comparisons.append({"mode": new["mode"], "seed": new["seed"],
                            "speedup": old["median_seconds"] / new["median_seconds"],
                            "exact_numerical_receipt_match": True,
                            "excluded_fields": ["seconds", "arithmetic.execution"]})
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--kernel", choices=("auto", "scalar", "avx2"), default="auto")
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    runtime = (root / "work").resolve()
    if not runtime.is_relative_to(root) or not output.is_relative_to(runtime):
        raise ValueError("benchmark receipts must stay under this checkout's work/")
    if output.exists():
        raise FileExistsError(output)
    result = measure(args.binary, args.kernel, args.repetitions)
    if args.baseline:
        result["comparison"] = compare(result, json.loads(args.baseline.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for case in result["cases"]:
        print(f"{case['mode']} seed {case['seed']}: {case['median_seconds']:.6f}s median")
    print(output)


if __name__ == "__main__":
    main()
