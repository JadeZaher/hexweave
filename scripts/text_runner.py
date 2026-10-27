"""Run one local bounded byte-model experiment under Windows resource admission."""

import argparse
import math
import time
import uuid
from pathlib import Path

from scripts.pulse_io import LOG_LIMIT, digest, event, publish
from scripts.text_child import run_child
from scripts.text_io import owned_parent, pinned_file, read_receipt
from scripts.text_resources import admission
from scripts.text_profile import evaluation_path, input_identity, validate_profile


def validate(steps: int, seed: int, timeout: float, anchor_interval: int,
             mode: str, backend: str, route_credit: str, profile: str, credit_diagnostics: str) -> None:
    validate_profile(profile, credit_diagnostics, mode)
    if type(steps) is not int or not 1 <= steps <= 1024:
        raise ValueError("steps must be 1..1024")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    if not math.isfinite(timeout) or not 1 <= timeout <= 120:
        raise ValueError("timeout must be finite and 1..120 seconds")
    if type(anchor_interval) is not int or not 1 <= anchor_interval <= 16:
        raise ValueError("anchor interval must be 1..16")
    if mode not in ("pulse", "anchor-only") or backend not in ("auto", "scalar", "avx2"):
        raise ValueError("unsupported text mode or kernel")
    if route_credit not in ("coupled", "route-to-u-detached"):
        raise ValueError("unsupported route credit")


def execute(root: Path, *, steps: int = 256, seed: int = 7, timeout: float = 30,
            anchor_interval: int = 4, mode: str = "pulse", backend: str = "auto",
            route_credit: str = "coupled", campaign_deadline: float | None = None,
            profile: str = "fixture", credit_diagnostics: str = "off") -> tuple[Path, dict]:
    validate(steps, seed, timeout, anchor_interval, mode, backend, route_credit, profile, credit_diagnostics)
    if campaign_deadline is not None and not math.isfinite(campaign_deadline):
        raise ValueError("campaign deadline must be finite")
    root = root.resolve(strict=True)
    parent = owned_parent(root)
    directory = parent / uuid.uuid4().hex
    directory.mkdir(exist_ok=False)
    report = {"status": "running", "resume": False, "automatic_retry": False,
              "native_chat_wakeup": False, "deadline_seconds": timeout,
              "log_detection_threshold_bytes": LOG_LIMIT, "requested_kernel": backend,
              "hard_limit_type": "Windows job committed memory, not a hard RSS cap",
              "rss_guard_type": "50 ms polling plus terminal peak-working-set query",
              "child": {}, "checkpoint_resumable": False}
    options = {"steps": steps, "seed": seed, "anchor_interval": anchor_interval,
               "mode": mode, "route_credit": route_credit, "profile": profile, "credit_diagnostics": credit_diagnostics}
    report["options"] = options
    deadline = time.monotonic() + timeout
    if campaign_deadline is not None:
        deadline = min(deadline, campaign_deadline)
    paths = {}
    hashes = {}
    try:
        event(directory, "run_started", **report)
        paths = {"binary": pinned_file(root, "work/target/release/hexweave.exe"),
                 "train": pinned_file(root, "datasets/tiny/train.txt"),
                 "valid": pinned_file(root, evaluation_path(profile))}
        if paths["binary"].stat().st_size > 128 * 1024**2 or any(
                paths[name].stat().st_size > 65_536 for name in ("train", "valid")):
            raise ValueError("input file exceeds bounded hash size")
        hashes = {name: digest(path) for name, path in paths.items()}
        report["input_sha256"] = hashes
        report["evaluation_input"] = input_identity(paths, hashes, profile)
        if (parent / "STOP").exists() or (directory / "STOP").exists():
            raise InterruptedError("STOP requested before launch")
        report["admission"] = admission()
        if time.monotonic() >= deadline:
            raise TimeoutError("deadline reached during admission")
        command = [str(paths["binary"]), "text", "--steps", str(steps), "--seed", str(seed),
                   "--anchor-interval", str(anchor_interval), "--mode", mode,
                   "--route-credit", route_credit,
                   "--profile", profile, "--credit-diagnostics", credit_diagnostics,
                   "--train", str(paths["train"]), "--valid", str(paths["valid"]),
                   "--work-dir", str(directory / "text")]
        event(directory, "launch_requested", options=options, hashes=hashes)
        run_child(root, command, directory, parent, deadline,
                  report["admission"]["host_cushion_bytes"], backend, report["child"])
        if time.monotonic() >= deadline:
            raise TimeoutError("whole-run deadline exceeded")
        if any(path.exists() for path in (parent / "STOP", directory / "STOP", directory / "text/STOP")):
            raise InterruptedError("STOP requested; terminal without retry")
        after = {name: digest(path) for name, path in paths.items()}
        report["input_sha256_after"] = after
        if after != hashes:
            raise RuntimeError("binary or dataset changed during experiment")
        report["result"] = read_receipt(directory / "text/result.json", options)
        if any(path.exists() for path in (parent / "STOP", directory / "STOP", directory / "text/STOP")):
            raise InterruptedError("STOP requested during receipt validation")
        if time.monotonic() >= deadline:
            raise TimeoutError("whole-run deadline exceeded during receipt validation")
        report["status"] = "completed"
    except (KeyboardInterrupt, InterruptedError) as error:
        report.update(status="stopped", error=str(error) or "keyboard interrupt")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if hashes:
            try:
                after = {name: digest(path) for name, path in paths.items()}
                report["input_sha256_after"] = after
                if after != hashes:
                    report.update(status="failed", identity_error="binary or dataset changed")
            except OSError as error:
                report.update(status="failed", identity_error=str(error))
        finalize(directory, parent, deadline, report)
    return directory, report


def terminal_guard(directory: Path, parent: Path, deadline: float, report: dict) -> bool:
    if report["status"] != "completed":
        return False
    if any(path.exists() for path in (parent / "STOP", directory / "STOP", directory / "text/STOP")):
        report.update(status="stopped", error="STOP requested during finalization")
    elif time.monotonic() >= deadline:
        report.update(status="failed", error="whole-run deadline exceeded during finalization")
    else:
        return False
    return True


def finalize(directory: Path, parent: Path, deadline: float, report: dict) -> None:
    terminal_guard(directory, parent, deadline, report)
    try:
        event(directory, "run_finished", status=report["status"])
    except OSError as error:
        report.update(status="failed", observer_error=str(error))
    publish(directory, report)
    if terminal_guard(directory, parent, deadline, report):
        publish(directory, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--anchor-interval", type=int, default=4)
    parser.add_argument("--mode", choices=("pulse", "anchor-only"), default="pulse")
    parser.add_argument("--backend", choices=("auto", "scalar", "avx2"), default="auto")
    parser.add_argument("--route-credit", choices=("coupled", "route-to-u-detached"), default="coupled")
    parser.add_argument("--profile", choices=("fixture", "t2-credit"), default="fixture")
    parser.add_argument("--credit-diagnostics", choices=("off", "on"), default="off")
    directory, report = execute(Path(__file__).resolve().parents[1], **vars(parser.parse_args()))
    print(f"{report['status']}: {directory / 'report.json'}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
