from backend.app.api.logs import get_logs
from backend.app.core.config import settings


def test_get_logs_returns_only_the_requested_tail(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "backend.log").write_text("first\nsecond\nthird\n", encoding="utf-8")
    monkeypatch.setattr(settings, "BACKEND_DIR", tmp_path)

    assert get_logs(lines=2) == ["second", "third"]


def test_get_logs_reports_when_log_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "BACKEND_DIR", tmp_path)

    assert get_logs(lines=100) == ["No log file found yet."]
