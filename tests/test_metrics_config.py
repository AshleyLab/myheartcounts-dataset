"""Tests for the Layer-1 MetricsConfig (configurable metric persistence)."""

from forecasting_evaluation.config import ForecastingEvalConfig, MetricsConfig


def test_defaults_are_full_sets():
    m = MetricsConfig()
    assert m.point_metrics == ["mae", "mse", "mase", "mase_all", "ql", "sql"]
    assert m.binary_metrics == ["auprc", "auroc", "f1"]
    assert m.combine_channels is True
    assert m.f1_threshold == 0.5


def test_attached_to_root_config():
    cfg = ForecastingEvalConfig()
    assert isinstance(cfg.metrics, MetricsConfig)
    # skill + ranking only need these two; both are present by default.
    assert "mae" in cfg.metrics.point_metrics
    assert "auprc" in cfg.metrics.binary_metrics


def test_binary_can_be_disabled():
    # Empty binary list is the documented way to skip the binary-metric pass.
    m = MetricsConfig(binary_metrics=[])
    assert m.binary_metrics == []


def test_minimal_skill_inputs():
    # The minimal persisted set sufficient for skill score + ranking.
    m = MetricsConfig(point_metrics=["mae"], binary_metrics=["auprc"])
    assert m.point_metrics == ["mae"]
    assert m.binary_metrics == ["auprc"]
