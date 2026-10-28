"""Bounded live-writer inventory; see causal_OWNERSHIP.md."""

import os
import re
import stat
from pathlib import Path


ENTRY_CAP = 8192
PUBLICATION_CAP = 131_072
REPARSE_POINT = 0x400


def regular(info, directory: bool) -> None:
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & REPARSE_POINT:
        raise ValueError("reparse entry in owned attempt inventory")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError("unexpected owned attempt entry type")


def vanished_owned_file(root: Path, path: Path, writer_pid: int | None, owner_pid: int | None = None) -> bool:
    owner_names = {"request.json", "report.json", "worker-request.json", "launch.json", "observer.json",
                   "worker-stop.json", "release.json", "release-ready.json", "release-invalid.json"}
    if owner_pid is not None and path.parent == root and path.name in {name + ".pending" for name in owner_names}:
        regular(root.lstat(), True)
        return True
    match = re.fullmatch(r"\.pending-([1-9][0-9]*)-(0|[1-9][0-9]*)", path.name)
    pending = (match is not None and len(match[2]) <= 20 and int(match[2]) <= 2**64 - 1
               and int(match[1]) == writer_pid)
    retained = re.fullmatch(r"(?:tile|shared|predictor|state|manifest)-[0-9a-f]{64}\.bin", path.name)
    commit = re.fullmatch(r"commit-([0-9]{20})-[0-9a-f]{64}\.json", path.name)
    committed = commit is not None and int(commit[1]) <= 2**64 - 1
    observation = re.fullmatch(r"\.child-([1-9][0-9]*)-assay-observation-[0-9a-f]{64}\.json", path.name)
    owned_observation = observation is not None and int(observation[1]) == writer_pid and path.parent == root
    result = re.fullmatch(r"\.child-([1-9][0-9]*)-result\.json", path.name)
    owned_result = owner_pid is not None and result is not None and int(result[1]) == writer_pid and path.parent == root
    if (writer_pid is None or not (owned_result or owned_observation or (path.parent == root / "persistence"
            and (pending or retained or committed)))):
        return False
    regular(root.lstat(), True)
    regular(path.parent.lstat(), True)
    marker = root / "persistence/OWNED"
    metadata = marker.lstat()
    regular(metadata, False)
    if metadata.st_size > 1024:
        raise ValueError("oversized owned persistence marker")
    with marker.open("rb") as source:
        contents = source.read(1025)
    if contents != f"causal-persistence-v1\n{root}\n".encode():
        raise ValueError("owned persistence marker mismatch")
    return True


def disk_bytes(directory: Path, live_writer_pid: int | None = None, live_owner_pid: int | None = None) -> int:
    if live_owner_pid is not None and (type(live_owner_pid) is not int or not 1 <= live_owner_pid <= 2**32 - 1):
        raise ValueError("invalid live owner identity")
    if live_writer_pid is not None and (type(live_writer_pid) is not int or not 1 <= live_writer_pid <= 2**32 - 1):
        raise ValueError("invalid live writer identity")
    root = directory.absolute()
    regular(root.lstat(), True)
    if root.resolve(strict=True) != root:
        raise ValueError("noncanonical owned inventory root")
    first, count, changed = scan(root, live_writer_pid, 0, True, live_owner_pid)
    if not changed:
        return first
    second, _, _ = scan(root, live_writer_pid, count, False, live_owner_pid)
    return max(first, second)


def scan(root: Path, live_writer_pid: int | None, count: int, allow_churn: bool, live_owner_pid: int | None = None):
    total, changed, pending = 0, False, [root]
    while pending:
        parent = pending.pop()
        regular(parent.lstat(), True)
        with os.scandir(parent) as entries:
            for entry in entries:
                count += 1
                path = Path(entry.path)
                if count > ENTRY_CAP or not path.is_relative_to(root):
                    raise ValueError("owned attempt inventory is unbounded or escaped")
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    if not allow_churn or not vanished_owned_file(root, path, live_writer_pid, live_owner_pid):
                        raise
                    total += PUBLICATION_CAP
                    changed = True
                    continue
                directory_entry = stat.S_ISDIR(metadata.st_mode)
                regular(metadata, directory_entry)
                if directory_entry:
                    pending.append(path)
                else:
                    total += metadata.st_size
    return total, count, changed
