"""Cross-platform process single-instance advisory lock helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass
class LockHandle:
    """Represents a held advisory lock."""

    file_obj: TextIO
    lock_path: Path
    platform: str


def _lock_path(app_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in app_name)
    return Path(tempfile.gettempdir()) / f"{safe_name}.lock"


def acquire_lock(app_name: str) -> LockHandle | None:
    """Acquire process lock for *app_name*.

    Returns a :class:`LockHandle` when lock was acquired, otherwise ``None``
    when another process already holds it.
    """

    lock_path = _lock_path(app_name)
    lock_file = open(lock_path, "a+", encoding="utf-8")

    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            platform = "nt"
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            platform = "posix"
    except OSError:
        lock_file.close()
        return None

    return LockHandle(file_obj=lock_file, lock_path=lock_path, platform=platform)


def release_lock(lock_handle: LockHandle | None) -> None:
    """Release previously acquired lock handle."""

    if lock_handle is None:
        return

    lock_file = lock_handle.file_obj
    try:
        if lock_handle.platform == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError:
        # Best-effort cleanup.
        pass
    finally:
        lock_file.close()
