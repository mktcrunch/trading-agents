"""Daily workflow dry-run mode."""
from src import config


def test_dry_run_mode_context():
    assert config.is_dry_run() is (config.DRY_RUN or False)
    with config.dry_run_mode(True):
        assert config.is_dry_run() is True
    assert config.is_dry_run() is (config.DRY_RUN or False)


def test_dry_run_mode_disabled():
    with config.dry_run_mode(False):
        assert config.is_dry_run() is config.DRY_RUN
