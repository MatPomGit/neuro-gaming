"""Testy replayu sesji i porównania wersji algorytmu kierunków."""

from pathlib import Path

from src.session_replay import SessionReplay, compare_direction_series


def _legacy_algorithm(event: dict) -> str:
    """Wersja bazowa algorytmu używana do porównań regresji."""
    metrics = event.get("metrics", {})
    af7 = metrics.get("AF7", {})
    af8 = metrics.get("AF8", {})
    beta_avg = (float(af7.get("beta", 0.0)) + float(af8.get("beta", 0.0))) / 2.0
    alpha_avg = (float(af7.get("alpha", 0.0)) + float(af8.get("alpha", 0.0))) / 2.0
    if beta_avg > 2.0:
        return "FORWARD"
    if alpha_avg > 2.0 and alpha_avg > beta_avg:
        return "BACKWARD"
    return "NONE"


def _new_algorithm(event: dict) -> str:
    """Nowsza wersja reguł z lateralizacją i progiem asymetrii."""
    metrics = event.get("metrics", {})
    af7 = metrics.get("AF7", {})
    af8 = metrics.get("AF8", {})
    beta_avg = (float(af7.get("beta", 0.0)) + float(af8.get("beta", 0.0))) / 2.0
    alpha_l = float(af7.get("alpha", 0.0))
    alpha_r = float(af8.get("alpha", 0.0))
    if beta_avg > 2.0:
        return "FORWARD"
    if alpha_l > alpha_r * 1.3:
        return "LEFT"
    if alpha_r > alpha_l * 1.3:
        return "RIGHT"
    return "NONE"


def test_compare_direction_series_on_same_replay_session(tmp_path: Path) -> None:
    """Porównuje kierunki z dwóch wersji algorytmu na tej samej sesji."""
    session_file = tmp_path / "case.session.jsonl"
    session_file.write_text(
        "\n".join(
            [
                '{"type":"session_header","format_version":"1.0","session_name":"cmp"}',
                '{"t":0.10,"type":"direction_decision","metrics":{"AF7":{"alpha":0.8,"beta":2.8},"AF8":{"alpha":0.7,"beta":2.7}}}',
                '{"t":0.20,"type":"direction_decision","metrics":{"AF7":{"alpha":3.4,"beta":0.9},"AF8":{"alpha":1.2,"beta":0.8}}}',
                '{"t":0.30,"type":"direction_decision","metrics":{"AF7":{"alpha":0.5,"beta":0.4},"AF8":{"alpha":0.6,"beta":0.3}}}',
            ]
        ),
        encoding="utf-8",
    )

    replay = SessionReplay(session_file)
    result = compare_direction_series(
        replay,
        algorithm_a=_legacy_algorithm,
        algorithm_b=_new_algorithm,
    )

    assert result["compared_windows"] == 3
    assert 0.0 <= result["agreement"] <= 1.0
    assert len(result["differences"]) == 1
    assert result["differences"][0]["t"] == 0.2
