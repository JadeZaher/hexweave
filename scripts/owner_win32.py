"""Win32 structures for the core owner's Job cap and memory sampling; see scripts/AGENTS.md."""

import ctypes as c
from ctypes import wintypes as w


class MemoryStatus(c.Structure):
    _fields_ = [("length", w.DWORD), ("load", w.DWORD)] + [
        (name, c.c_ulonglong) for name in (
            "total_physical", "available_physical", "total_pagefile",
            "available_pagefile", "total_virtual", "available_virtual", "extended")]


class MemoryCounters(c.Structure):
    _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [
        (name, c.c_size_t) for name in (
            "peak_working_set", "working_set", "peak_paged", "paged",
            "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]


class BasicLimits(c.Structure):
    _fields_ = [("process_time", c.c_longlong), ("job_time", c.c_longlong),
                ("flags", w.DWORD), ("minimum_working_set", c.c_size_t),
                ("maximum_working_set", c.c_size_t), ("active_processes", w.DWORD),
                ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class IoCounters(c.Structure):
    _fields_ = [(name, c.c_ulonglong) for name in (
        "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(c.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                ("peak_process_memory", c.c_size_t), ("peak_job_memory", c.c_size_t)]


class ThreadEntry(c.Structure):
    _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("thread_id", w.DWORD),
                ("owner_pid", w.DWORD), ("base_priority", w.LONG),
                ("delta_priority", w.LONG), ("flags", w.DWORD)]


def bind(library, name, result, arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = arguments
    return function


def kernel():
    library = c.WinDLL("kernel32", use_last_error=True)
    bind(library, "GlobalMemoryStatusEx", w.BOOL, [c.POINTER(MemoryStatus)])
    bind(library, "CreateJobObjectW", w.HANDLE, [c.c_void_p, w.LPCWSTR])
    bind(library, "SetInformationJobObject", w.BOOL, [w.HANDLE, c.c_int, c.c_void_p, w.DWORD])
    bind(library, "AssignProcessToJobObject", w.BOOL, [w.HANDLE, w.HANDLE])
    bind(library, "CloseHandle", w.BOOL, [w.HANDLE])
    bind(library, "CreateToolhelp32Snapshot", w.HANDLE, [w.DWORD, w.DWORD])
    bind(library, "Thread32First", w.BOOL, [w.HANDLE, c.POINTER(ThreadEntry)])
    bind(library, "Thread32Next", w.BOOL, [w.HANDLE, c.POINTER(ThreadEntry)])
    bind(library, "OpenThread", w.HANDLE, [w.DWORD, w.BOOL, w.DWORD])
    bind(library, "ResumeThread", w.DWORD, [w.HANDLE])
    return library
