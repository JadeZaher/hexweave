"""Physical admission, sampled RSS guard and hard Windows job commit limits."""

import ctypes as c
import os
from ctypes import wintypes as w

from scripts.text_win32 import ExtendedLimits, MemoryCounters, MemoryStatus, ThreadEntry, bind, kernel


GIB = 1024**3
CHILD_LIMIT = 512 * 1024**2


def required_cushion(total: int) -> int:
    return max(2 * GIB, int(total * 0.15))


def memory_status() -> dict:
    if os.name != "nt":
        raise RuntimeError("text resource admission requires Windows")
    state = MemoryStatus()
    state.length = c.sizeof(state)
    if not kernel().GlobalMemoryStatusEx(c.byref(state)):
        raise c.WinError(c.get_last_error())
    return {"total_physical_bytes": state.total_physical,
            "available_physical_bytes": state.available_physical}


def admission() -> dict:
    state = memory_status()
    cushion = required_cushion(state["total_physical_bytes"])
    if state["available_physical_bytes"] < cushion + CHILD_LIMIT:
        raise MemoryError("physical admission needs adaptive host cushion plus 512 MiB")
    return state | {"host_cushion_bytes": cushion, "child_commit_limit_bytes": CHILD_LIMIT}


def process_memory(process) -> dict:
    library = c.WinDLL("psapi", use_last_error=True)
    function = bind(library, "GetProcessMemoryInfo", w.BOOL,
                    [w.HANDLE, c.POINTER(MemoryCounters), w.DWORD])
    counters = MemoryCounters()
    counters.cb = c.sizeof(counters)
    if not function(int(process._handle), c.byref(counters), c.sizeof(counters)):
        raise c.WinError(c.get_last_error())
    return {"working_set_bytes": counters.working_set,
            "peak_working_set_bytes": counters.peak_working_set,
            "pagefile_bytes": counters.pagefile, "peak_pagefile_bytes": counters.peak_pagefile}


class Job:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("hard job limits require Windows")
        self.api = kernel()
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000 | 0x0008 | 0x0100 | 0x0200
        limits.basic.active_processes = 1
        limits.process_memory = CHILD_LIMIT
        limits.job_memory = CHILD_LIMIT
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.close()
            raise error

    def assign(self, process) -> None:
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise c.WinError(c.get_last_error())

    def close(self) -> None:
        if self.handle:
            if not self.api.CloseHandle(self.handle):
                raise c.WinError(c.get_last_error())
            self.handle = None


def resume_suspended(process) -> None:
    api = kernel()
    snapshot = api.CreateToolhelp32Snapshot(0x00000004, 0)
    if snapshot == c.c_void_p(-1).value:
        raise c.WinError(c.get_last_error())
    try:
        entry = ThreadEntry()
        entry.size = c.sizeof(entry)
        found = api.Thread32First(snapshot, c.byref(entry))
        while found:
            if entry.owner_pid == process.pid:
                _resume_thread(api, entry.thread_id)
                return
            found = api.Thread32Next(snapshot, c.byref(entry))
        raise RuntimeError("suspended owned process has no primary thread")
    finally:
        api.CloseHandle(snapshot)


def _resume_thread(api, thread_id: int) -> None:
    handle = api.OpenThread(0x0002, False, thread_id)
    if not handle:
        raise c.WinError(c.get_last_error())
    try:
        count = api.ResumeThread(handle)
        if count != 1:
            raise RuntimeError(f"unexpected initial suspend count: {count}")
    finally:
        api.CloseHandle(handle)
