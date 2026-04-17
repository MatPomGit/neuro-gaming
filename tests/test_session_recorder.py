"""Testy rejestratora sesji (CSV i replay)."""

from pathlib import Path

from src.session_recorder import SessionRecorder


def test_session_recorder_records_and_snapshots() -> None:
    """Sprawdza, czy próbki są zliczane i widoczne w snapshot."""
    recorder = SessionRecorder()
    recorder.start(now_monotonic=10.0, session_name="demo")
    recorder.record_sample(
        now_monotonic=10.5,
        direction="FORWARD",
        connected=True,
        motion_artifact=False,
        metrics={"AF7": {"alpha": 1.0, "beta": 2.0}, "AF8": {"alpha": 1.2, "beta": 2.4}},
        signal_quality={"AF7": 0.8, "AF8": 0.6},
    )

    snap = recorder.snapshot()
    assert snap["session_name"] == "demo"
    assert snap["samples"] == 1
    assert snap["duration"] == 0.5


def test_session_recorder_exports_csv_and_replay(tmp_path: Path) -> None:
    """Weryfikuje eksport CSV oraz dane replay w kolejności czasu."""
    recorder = SessionRecorder()
    recorder.start(now_monotonic=0.0, session_name="csv_case")
    recorder.record_sample(
        now_monotonic=0.1,
        direction="LEFT",
        connected=False,
        motion_artifact=True,
        metrics={"AF7": {"alpha": 0.5, "beta": 0.2}, "AF8": {"alpha": 0.4, "beta": 0.1}},
        signal_quality={"TP9": 0.3, "TP10": 0.5},
    )
    recorder.stop()

    csv_path = recorder.export_csv(tmp_path / "session.csv")
    content = csv_path.read_text(encoding="utf-8")
    assert "relative_time,direction,connected" in content
    assert "LEFT" in content

    replay = recorder.replay_data()
    assert len(replay) == 1
    assert replay[0]["direction"] == "LEFT"
