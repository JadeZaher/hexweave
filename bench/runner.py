"""Record bounded deployment measurements; see bench/AGENTS.md."""

import datetime
import http.client
import statistics
import time
from pathlib import Path

from .client import CAPABILITIES, complete
from .files import exclusive_output, write_record
from .manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def _requests(rows, budget):
    for warmup in range(budget["warmups"]):
        yield "warmup", warmup, rows[0]
    for repetition in range(budget["repetitions"]):
        for row in rows:
            yield "measured", repetition, row


def _summary(status, measured, elapsed, requested):
    latencies = [record["end_to_end_seconds"] for record in measured]
    scores = [record["exact_match"] for record in measured if record["exact_match"] is not None]
    total_bytes = sum(record["output_utf8_bytes"] for record in measured)
    request_time = sum(latencies)
    return {"record": "summary", "status": status, "complete": status == "completed",
            "elapsed_seconds": elapsed, "requested_measured_requests": requested,
            "completed_measured_requests": len(measured),
            "mean_end_to_end_seconds": statistics.mean(latencies) if latencies else None,
            "median_end_to_end_seconds": statistics.median(latencies) if latencies else None,
            "p95_end_to_end_seconds": None,
            "p95_note": "Not estimated by the scaffold; inspect raw request observations",
            "output_bytes_per_end_to_end_second": total_bytes / request_time if request_time else None,
            "exact_match": statistics.mean(scores) if scores else None,
            "exact_match_count": len(scores), "capabilities": CAPABILITIES,
            "artifact_endpoint_binding_verified": False, "leaderboard_eligible": False}


def run(manifest_path, output_path, *, root=ROOT, requester=complete):
    """Run sequential requests against an existing endpoint and persist failures."""
    snapshot = load_manifest(manifest_path)
    manifest, rows = snapshot["manifest"], snapshot["rows"]
    budget = manifest["budget"]
    measured, status = [], "completed"
    with exclusive_output(root, output_path) as handle:
        started = time.perf_counter()
        write_record(handle, {"record": "manifest", "manifest": manifest,
                              "manifest_sha256": snapshot["manifest_sha256"],
                              "dataset_sha256": manifest["dataset"]["sha256"],
                              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              "capabilities": CAPABILITIES, "artifact_endpoint_binding_verified": False,
                              "identity_note": "Local artifact hashes checked; server artifact binding is unverified"})
        for phase, repetition, row in _requests(rows, budget):
            remaining = budget["max_wall_seconds"] - (time.perf_counter() - started)
            record = {"record": "example", "phase": phase, "repetition": repetition,
                      "id": row["id"], "prompt": row["prompt"], "expected": row["expected"],
                      "prompt_utf8_bytes": len(row["prompt"].encode("utf-8"))}
            try:
                if remaining <= 0:
                    raise TimeoutError("Run wall-time budget exhausted")
                result = requester(manifest, row["prompt"], min(budget["timeout_seconds"], remaining))
                record.update(result, failure=None)
                record["exact_match"] = (result["output"] == row["expected"]
                                         if row["expected"] is not None else None)
                write_record(handle, record)
                if phase == "measured":
                    measured.append(record)
            except (OSError, ValueError, http.client.HTTPException) as error:
                status = "failed"
                record.update(output=None, finish_reason=None, truncated=None,
                              failure={"type": type(error).__name__, "message": str(error)[:512]})
                write_record(handle, record)
                break
            except KeyboardInterrupt:
                status = "stopped"
                record.update(output=None, finish_reason=None, truncated=None,
                              failure={"type": "KeyboardInterrupt", "message": "User interrupted run"})
                write_record(handle, record)
                break
        summary = _summary(status, measured, time.perf_counter() - started,
                           budget["examples"] * budget["repetitions"])
        write_record(handle, summary)
    return summary
