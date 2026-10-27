"""Exact identities and sampled owner/child admission; see causal_OWNERSHIP.md."""

import ctypes as c
import os
import shutil
from ctypes import wintypes as w
from types import SimpleNamespace

from scripts.text_resources import CHILD_LIMIT, Job, memory_status, process_memory, required_cushion, resume_suspended
from scripts.text_win32 import bind, kernel


OWNER_LIMIT = 256 * 1024**2
SAMPLED_SUM_LIMIT = OWNER_LIMIT + CHILD_LIMIT
ELASTIC_RUNTIME_FLOOR = 256 * 1024**2
MODES = ("strict", "elastic")


def admission_mode() -> str:
    mode = os.environ.get("HEXWEAVE_CAUSAL_ADMISSION", "strict")
    if mode not in MODES:
        raise ValueError("unknown HEXWEAVE_CAUSAL_ADMISSION mode")
    return mode


def admission_floor(total: int, overhead: int, mode: str) -> int:
    """Start headroom floor; see scripts/AGENTS.md §admission modes."""
    if mode == "elastic":
        return ELASTIC_RUNTIME_FLOOR + CHILD_LIMIT
    return required_cushion(total) + CHILD_LIMIT + max(0, OWNER_LIMIT - overhead)


def runtime_floor(total: int, mode: str) -> int:
    """In-run sampled availability floor enforced by sample()."""
    return ELASTIC_RUNTIME_FLOOR if mode == "elastic" else required_cushion(total)


def owner_process():
    if os.name != "nt":
        raise RuntimeError("causal ownership requires Windows")
    api = kernel()
    current = bind(api, "GetCurrentProcess", w.HANDLE, [])()
    priority = bind(api, "SetPriorityClass", w.BOOL, [w.HANDLE, w.DWORD])
    if not priority(current, 0x00004000):
        raise c.WinError(c.get_last_error())
    return SimpleNamespace(pid=os.getpid(), _handle=current)


def process_identity(process) -> dict:
    times = [w.FILETIME() for _ in range(4)]
    function = bind(kernel(), "GetProcessTimes", w.BOOL, [w.HANDLE] + [c.POINTER(w.FILETIME)] * 4)
    if not function(int(process._handle), *(c.byref(value) for value in times)):
        raise c.WinError(c.get_last_error())
    return {"pid": process.pid, "creation_filetime": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}


def alive(process) -> bool:
    code = w.DWORD()
    function = bind(kernel(), "GetExitCodeProcess", w.BOOL, [w.HANDLE, c.POINTER(w.DWORD)])
    if not function(int(process._handle), c.byref(code)):
        raise c.WinError(c.get_last_error())
    return code.value == 259


def driver_process():
    pid = int(os.environ["HEXWEAVE_CAUSAL_DRIVER_PID"])
    created = int(os.environ["HEXWEAVE_CAUSAL_DRIVER_CREATED"])
    function = bind(kernel(), "OpenProcess", w.HANDLE, [w.DWORD, w.BOOL, w.DWORD])
    handle = function(0x0400 | 0x0010, False, pid)
    if not handle:
        raise c.WinError(c.get_last_error())
    process = SimpleNamespace(pid=pid, _handle=handle)
    if process_identity(process) != {"pid": pid, "creation_filetime": created} or not alive(process):
        close_driver(process)
        raise ValueError("external causal driver identity is not live")
    return process


def close_driver(driver) -> None:
    if not kernel().CloseHandle(int(driver._handle)):
        raise c.WinError(c.get_last_error())


def identity_absent(identity: dict) -> bool:
    if (not isinstance(identity, dict) or set(identity) != {"pid", "creation_filetime"}
            or type(identity["pid"]) is not int or not 0 < identity["pid"] < 2**32
            or type(identity["creation_filetime"]) is not int or not 0 < identity["creation_filetime"] < 2**64):
        raise ValueError("invalid exact process identity")
    function = bind(kernel(), "OpenProcess", w.HANDLE, [w.DWORD, w.BOOL, w.DWORD])
    handle = function(0x1000, False, identity["pid"])
    if not handle:
        error = c.get_last_error()
        if error == 87:
            return True
        raise c.WinError(error)
    process = SimpleNamespace(pid=identity["pid"], _handle=handle)
    try:
        return process_identity(process) != identity or not alive(process)
    finally:
        close_driver(process)


def admission(owner, directory, disk_allowance: int) -> dict:
    mode = admission_mode()
    state = memory_status()
    memory = process_memory(owner)
    driver_memory = process_memory(owner.driver)
    overhead = memory["working_set_bytes"] + driver_memory["working_set_bytes"]
    floor = admission_floor(state["total_physical_bytes"], overhead, mode)
    if overhead > OWNER_LIMIT:
        raise MemoryError(f"owner/driver sampled RSS {overhead} exceeds {OWNER_LIMIT} at admission")
    if not alive(owner.driver):
        raise MemoryError("external driver is not live at admission")
    if state["available_physical_bytes"] < floor:
        raise MemoryError(f"{mode} admission: available {state['available_physical_bytes']} below start floor {floor}")
    free = shutil.disk_usage(directory).free
    if free < disk_allowance + 256 * 1024**2:
        raise OSError("insufficient free space for declared attempt and disk cushion")
    return state | {"owner": process_identity(owner), "owner_memory": memory,
                    "driver": process_identity(owner.driver), "driver_memory": driver_memory,
                    "admission_mode": mode, "admission_floor_bytes": floor,
                    "policy_cushion_bytes": required_cushion(state["total_physical_bytes"]),
                    "host_cushion_bytes": runtime_floor(state["total_physical_bytes"], mode),
                    "disk_free_bytes": free,
                    "disk_allowance_bytes": disk_allowance, "child_commit_limit_bytes": CHILD_LIMIT,
                    "driver_owner_sampled_rss_limit_bytes": OWNER_LIMIT, "sampled_sum_limit_bytes": SAMPLED_SUM_LIMIT,
                    "scope": "exact external driver, Python owner and Job-contained child; adjacent RSS samples, not instantaneous tree enforcement"}


def sample(owner, child, report: dict, cushion: int) -> None:
    current_owner, current_child = process_memory(owner), process_memory(child)
    current_driver = process_memory(owner.driver)
    if not alive(owner.driver):
        raise RuntimeError("external driver exited before owner release")
    overhead = current_owner["working_set_bytes"] + current_driver["working_set_bytes"]
    total = overhead + current_child["working_set_bytes"]
    report.update(driver_memory=current_driver, owner_memory=current_owner, child_memory=current_child,
                  peak_sampled_driver_owner_child_rss=max(total, report.get("peak_sampled_driver_owner_child_rss", 0)))
    if (overhead > OWNER_LIMIT or current_child["working_set_bytes"] > CHILD_LIMIT
            or total > SAMPLED_SUM_LIMIT):
        raise MemoryError(f"sampled causal owner/child RSS bound exceeded: driver+owner {overhead},"
                          f" child {current_child['working_set_bytes']}, sum {total}")
    available = memory_status()["available_physical_bytes"]
    if available < cushion:
        raise MemoryError(f"adaptive physical cushion reached: available {available} below runtime floor {cushion}")
