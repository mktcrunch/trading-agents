from src.discovery.catalog import is_probeable_schema, probe_lookback_days


def test_excludes_sub_15m_ohlcv_schemas():
    assert not is_probeable_schema("ohlcv-1s")
    assert not is_probeable_schema("ohlcv-1m")
    assert not is_probeable_schema("ohlcv-5m")


def test_allows_coarse_ohlcv_schemas():
    assert is_probeable_schema("ohlcv-1d")
    assert is_probeable_schema("ohlcv-1h")
    assert is_probeable_schema("ohlcv-15m")
    assert is_probeable_schema("statistics")


def test_schema_specific_lookback_days():
    assert probe_lookback_days("ohlcv-1d") == 90
    assert probe_lookback_days("ohlcv-1h") == 45
    assert probe_lookback_days("ohlcv-15m") == 20
    assert probe_lookback_days("ohlcv-30m") == 20
