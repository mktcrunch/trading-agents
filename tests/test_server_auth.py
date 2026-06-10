"""Cloud Run job route auth (dashboard stays public)."""
from unittest.mock import MagicMock, patch

from server import _check_auth


def _handler(secret_header: str | None = None) -> MagicMock:
    h = MagicMock()
    h.headers.get.return_value = secret_header
    return h


def test_job_auth_fails_when_secret_unset():
    with patch("server.config.SCHEDULER_SECRET", ""):
        assert _check_auth(_handler("anything")) is False


def test_job_auth_fails_on_wrong_header():
    with patch("server.config.SCHEDULER_SECRET", "correct"):
        assert _check_auth(_handler("wrong")) is False
        assert _check_auth(_handler(None)) is False


def test_job_auth_passes_on_matching_header():
    with patch("server.config.SCHEDULER_SECRET", "correct"):
        assert _check_auth(_handler("correct")) is True
