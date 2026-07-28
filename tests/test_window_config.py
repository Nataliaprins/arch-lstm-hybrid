"""tests/test_window_config.py — Section 5: single project-wide data.window."""
import pytest

from src.tuning.tune_and_train import _get_window_size


def _base_cfg():
    return {
        "data": {"window": 22, "input_scaling": "unconditional"},
        "series": [{"name": "BTC-USD"}, {"name": "ETH-USD"}],
        "hyperparameter_search": {"lstm_units": [16]},
    }


def test_reads_data_window():
    assert _get_window_size(_base_cfg()) == 22


def test_rejects_legacy_top_level_window_size():
    cfg = _base_cfg()
    cfg["window_size"] = 22
    with pytest.raises(ValueError, match="window_size"):
        _get_window_size(cfg)


def test_missing_data_window_raises():
    cfg = _base_cfg()
    del cfg["data"]["window"]
    with pytest.raises(ValueError, match="data.window"):
        _get_window_size(cfg)


@pytest.mark.parametrize("key", ["window", "window_size"])
def test_aborts_on_per_series_override(key):
    cfg = _base_cfg()
    cfg["series"][1][key] = 100   # per-market override on ETH-USD
    with pytest.raises(ValueError, match="ETH-USD"):
        _get_window_size(cfg)


@pytest.mark.parametrize("key", ["window", "window_size", "window_values"])
def test_aborts_on_hyperparameter_search_window(key):
    cfg = _base_cfg()
    cfg["hyperparameter_search"][key] = [1, 2, 100]
    with pytest.raises(ValueError, match="hyperparameter_search"):
        _get_window_size(cfg)
