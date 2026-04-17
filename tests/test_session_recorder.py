"""Testy rejestratora sesji (CSV, JSONL, raport)."""

from pathlib import Path

from src.session_recorder import SessionRecorder


def test_session_recorder_records_and_snapshots() -> None:
    """Sprawdza, czy próbki są zliczane i widoczne w snapshot."""
    recorder = SessionRecorder()
    recorder.start(now_monotonic=10.0, session_name="demo")
    recorder.record_eeg_frame(now_monotonic=10.2, channel="AF7", samples=[1.0, 1.2, 1.1])
    recorder.record_sample(
        now_monotonic=10.5,
        direction="FORWARD",
        connected=True,
        motion_artifact=False,
        metrics={"AF7": {"alpha": 1.0, "beta": 2.0}, "AF8": {"alpha": 1.2, "beta": 2.4}},
        signal_quality={"AF7": 0.8, "AF8": 0.6},
        confidence=0.75,
        rejected_window=False,
    )

    snap = recorder.snapshot()
    assert snap["session_name"] == "demo"
    assert snap["samples"] == 1
    assert snap["duration"] == 0.5


def test_session_recorder_exports_files_and_report(tmp_path: Path) -> None:
    """Weryfikuje eksport CSV, pełnej sesji oraz raport po sesji."""
    recorder = SessionRecorder()
    recorder.start(
        now_monotonic=0.0,
        session_name="csv_case",
        metadata={"device_model": "Muse S", "app_version": "1.2.0"},
    )
    recorder.record_eeg_frame(now_monotonic=0.05, channel="AF7", samples=[0.1, 0.2])
    recorder.record_imu_frame(now_monotonic=0.06, sensor="accelerometer", samples=[[0.0, 0.1, 0.2]])
    recorder.record_ppg_frame(now_monotonic=0.07, channel="PPG_IR", samples=[9.0, 9.1])
    recorder.record_control_event(now_monotonic=0.08, event_name="key_down", payload={"key": "up"})
    recorder.record_sample(
        now_monotonic=0.1,
        direction="LEFT",
        connected=True,
        motion_artifact=True,
        metrics={"AF7": {"alpha": 0.5, "beta": 0.2}, "AF8": {"alpha": 0.4, "beta": 0.1}},
        signal_quality={"TP9": 0.3, "TP10": 0.5},
        confidence=0.4,
        rejected_window=True,
    )
    recorder.stop()

    csv_path = recorder.export_csv(tmp_path / "session.csv")
    content = csv_path.read_text(encoding="utf-8")
    assert "relative_time,direction,connected,motion_artifact" in content
    assert "LEFT" in content

    extended_csv_path = recorder.export_csv_extended(tmp_path / "session.extended.csv")
    extended_content = extended_csv_path.read_text(encoding="utf-8")
    assert "decision_latency_ms" in extended_content

    session_path = recorder.export_session(tmp_path / "session.session.jsonl")
    session_content = session_path.read_text(encoding="utf-8")
    assert "session_header" in session_content
    assert '"type": "eeg"' in session_content
    assert '"type": "imu"' in session_content
    assert '"type": "ppg"' in session_content

    report_path = recorder.export_report(tmp_path / "session.report.json")
    report_content = report_path.read_text(encoding="utf-8")
    assert "average_latency_ms" in report_content
    assert "rejected_windows" in report_content

    replay = recorder.replay_data()
    assert len(replay) == 1
    assert replay[0]["direction"] == "LEFT"
