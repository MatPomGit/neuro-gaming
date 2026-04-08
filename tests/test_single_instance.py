import sys
import types
from pathlib import Path
import builtins

from src import single_instance


class DummyFile:
    def __init__(self):
        self.closed = False
        self._fd = 123
        self.seek_calls = []

    def fileno(self):
        return self._fd

    def seek(self, pos):
        self.seek_calls.append(pos)

    def close(self):
        self.closed = True


def test_acquire_lock_posix_success(monkeypatch, tmp_path):
    dummy_file = DummyFile()
    flock_calls = []

    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: dummy_file)
    monkeypatch.setattr(single_instance, "_lock_path", lambda app_name: tmp_path / "ng.lock")

    fcntl_mock = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda fd, flags: flock_calls.append((fd, flags)))
    monkeypatch.setitem(sys.modules, "fcntl", fcntl_mock)
    monkeypatch.setattr(single_instance.os, "name", "posix")

    lock = single_instance.acquire_lock("neuro_gaming")

    assert lock is not None
    assert lock.file_obj is dummy_file
    assert lock.lock_path == tmp_path / "ng.lock"
    assert flock_calls == [(123, 3)]


def test_acquire_lock_posix_busy_returns_none(monkeypatch):
    dummy_file = DummyFile()

    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: dummy_file)

    def _raise_busy(_fd, _flags):
        raise OSError("busy")

    fcntl_mock = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=_raise_busy)
    monkeypatch.setitem(sys.modules, "fcntl", fcntl_mock)
    monkeypatch.setattr(single_instance.os, "name", "posix")

    lock = single_instance.acquire_lock("neuro_gaming")

    assert lock is None
    assert dummy_file.closed


def test_release_lock_posix(monkeypatch):
    dummy_file = DummyFile()
    unlock_calls = []

    fcntl_mock = types.SimpleNamespace(LOCK_UN=8, flock=lambda fd, flags: unlock_calls.append((fd, flags)))
    monkeypatch.setitem(sys.modules, "fcntl", fcntl_mock)

    handle = single_instance.LockHandle(file_obj=dummy_file, lock_path=Path("/tmp/ng.lock"), platform="posix")
    single_instance.release_lock(handle)

    assert unlock_calls == [(123, 8)]
    assert dummy_file.closed


def test_acquire_lock_windows_success(monkeypatch, tmp_path):
    dummy_file = DummyFile()
    locking_calls = []

    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: dummy_file)
    monkeypatch.setattr(single_instance, "_lock_path", lambda app_name: tmp_path / "ng.lock")

    msvcrt_mock = types.SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=lambda fd, mode, size: locking_calls.append((fd, mode, size)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt_mock)
    monkeypatch.setattr(single_instance.os, "name", "nt")

    lock = single_instance.acquire_lock("neuro_gaming")

    assert lock is not None
    assert locking_calls == [(123, 1, 1)]
    assert dummy_file.seek_calls == [0]


def test_release_lock_windows(monkeypatch):
    dummy_file = DummyFile()
    locking_calls = []

    msvcrt_mock = types.SimpleNamespace(
        LK_UNLCK=2,
        locking=lambda fd, mode, size: locking_calls.append((fd, mode, size)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt_mock)

    handle = single_instance.LockHandle(file_obj=dummy_file, lock_path=Path("C:/temp/ng.lock"), platform="nt")
    single_instance.release_lock(handle)

    assert locking_calls == [(123, 2, 1)]
    assert dummy_file.seek_calls == [0]
    assert dummy_file.closed


def test_release_lock_accepts_none():
    single_instance.release_lock(None)


def test_lock_path_sanitizes_app_name(monkeypatch):
    monkeypatch.setattr(single_instance.tempfile, "gettempdir", lambda: "/tmp")
    lock_path = single_instance._lock_path("Neuro Gaming:App")
    assert str(lock_path).endswith("Neuro_Gaming_App.lock")
