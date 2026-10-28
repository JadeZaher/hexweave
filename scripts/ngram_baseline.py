"""Byte-level interpolated Witten-Bell n-gram baseline for the core path's held-out bits/byte metric."""

import argparse
import ctypes
import json
import math
import os
import time
from ctypes import wintypes
from pathlib import Path

START = 0x02
HELDOUT_SPANS = 16
HELDOUT_SPAN = 128
ORDER_FIVE_MEMORY_LIMIT_BYTES = 2_000_000_000
MEMORY_CHECK_EVERY = 250_000


class _MemoryCounters(ctypes.Structure):
    """Layout of Win32 PROCESS_MEMORY_COUNTERS, used only by peak_working_set_bytes."""

    _fields_ = [
        ("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_size", ctypes.c_size_t), ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t), ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t), ("quota_nonpaged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def peak_working_set_bytes() -> int | None:
    """Current process's Windows peak working-set size in bytes, or None off Windows / on failure."""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _MemoryCounters(cb=ctypes.sizeof(_MemoryCounters))
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.peak_working_set_size)
    except OSError:
        return None


def _build_single_order(stream: bytes, order: int, memory_limit: int | None) -> dict | None:
    """Count one order's (context, next-byte) pairs; abort past memory_limit peak working set."""
    table: dict[bytes, dict[int, int]] = {}
    for i in range(max(0, order - 2), len(stream) - 1):
        context = stream[i + 2 - order:i + 1]
        target = stream[i + 1]
        counts = table.get(context)
        if counts is None:
            table[context] = counts = {}
        counts[target] = counts.get(target, 0) + 1
        if memory_limit is not None and i % MEMORY_CHECK_EVERY == 0:
            used = peak_working_set_bytes()
            if used is not None and used > memory_limit:
                return None
    return table


def build_tables(stream: bytes, max_order: int, memory_limit: int | None = None) -> dict[int, dict]:
    """Count tables for every order 1..max_order; stops (keeping lower orders) if order 5+ exceeds memory_limit."""
    tables: dict[int, dict] = {}
    for order in range(1, max_order + 1):
        guard = memory_limit if order >= 5 else None
        table = _build_single_order(stream, order, guard)
        if table is None:
            break
        tables[order] = table
    return tables


def probability(byte_value: int, tail: bytes, order: int, tables: dict) -> float:
    """Interpolated Witten-Bell probability of byte_value after tail, backing off order to uniform."""
    if order == 0:
        return 1.0 / 256
    context = tail[-(order - 1):] if order > 1 else b""
    counts = tables.get(order, {}).get(context)
    lower = probability(byte_value, tail, order - 1, tables)
    if counts is None:
        return lower
    total = sum(counts.values())
    weight = total / (total + len(counts))
    return weight * (counts.get(byte_value, 0) / total) + (1 - weight) * lower


def score(tables: dict, max_order: int, preceding: bytes, targets: bytes) -> tuple[float, int]:
    """Teacher-forced sum of -log2 probability over targets, sliding the context after each byte."""
    tail = preceding[-(max_order - 1):] if max_order > 1 else b""
    total = 0.0
    for byte_value in targets:
        total += -math.log2(probability(byte_value, tail, max_order, tables))
        tail = (tail + bytes((byte_value,)))[-(max_order - 1):] if max_order > 1 else b""
    return total, len(targets)


def heldout_bits_per_byte(tables: dict, max_order: int, heldout: bytes) -> tuple[float, int]:
    """Mean -log2 probability per target over the core path's 16-span, 128-target held-out protocol."""
    stream = bytes((START,)) + heldout
    length = len(heldout)
    stride = length // HELDOUT_SPANS
    tail_width = max_order - 1
    total_bits, total_targets = 0.0, 0
    for span in range(HELDOUT_SPANS):
        begin = span * stride
        end = min(begin + HELDOUT_SPAN, length)
        preceding = stream[max(0, begin + 1 - tail_width):begin + 1]
        bits, count = score(tables, max_order, preceding, heldout[begin:end])
        total_bits += bits
        total_targets += count
    return total_bits / total_targets, total_targets


def run(train: bytes, heldout: bytes, train_bytes_values: list[int], max_order: int) -> list[dict]:
    """Build tables and score every (train_bytes, order) combination requested."""
    results = []
    for train_bytes in train_bytes_values:
        started = time.perf_counter()
        stream = bytes((START,)) + train[:train_bytes]
        tables = build_tables(stream, max_order, ORDER_FIVE_MEMORY_LIMIT_BYTES)
        achieved = max(tables) if tables else 0
        elapsed = time.perf_counter() - started
        for order in range(1, achieved + 1):
            bits, targets = heldout_bits_per_byte(tables, order, heldout)
            results.append({"train_bytes": train_bytes, "order": order,
                             "bits_per_byte": bits, "heldout_targets": targets,
                             "build_seconds": elapsed})
        if achieved < max_order:
            results.append({"train_bytes": train_bytes, "order": max_order,
                             "bits_per_byte": None, "skipped_reason": "memory_limit"})
        print(f"train_bytes={train_bytes} built orders 1..{achieved} in {elapsed:.1f}s", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    """Parse the CLI contract: --train, --heldout, --orders, repeatable --train-bytes, --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--orders", type=int, default=4)
    parser.add_argument("--train-bytes", type=int, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.orders < 1:
        raise SystemExit("--orders must be >= 1")
    started = time.perf_counter()
    train_data = arguments.train.read_bytes()
    heldout_data = arguments.heldout.read_bytes()
    for train_bytes in arguments.train_bytes:
        if train_bytes > len(train_data):
            raise SystemExit(f"--train-bytes {train_bytes} exceeds train file length {len(train_data)}")
    results = run(train_data, heldout_data, arguments.train_bytes, arguments.orders)
    payload = {
        "schema": 1, "train_path": str(arguments.train), "heldout_path": str(arguments.heldout),
        "heldout_bytes": len(heldout_data), "requested_orders": arguments.orders,
        "runtime_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": peak_working_set_bytes(),
        "runs": results,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload["runs"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
