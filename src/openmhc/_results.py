"""Result containers for MHC-Benchmark evaluations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionResults:
    """Results from health-prediction evaluation.

    Attributes:
        records: List of per-task metric records. Each record is a dict with
            keys: task, task_type, classifier, metric, value, n_train, n_test.
        global_score: Primary ranking metric (mean test AUROC across binary
            tasks).
    """

    records: list[dict] = field(repr=False)
    global_score: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns:
            DataFrame with columns: task, task_type, classifier, metric,
            value, n_train, n_test.
        """
        return pd.DataFrame(self.records)

    def to_csv(self, path: str | Path) -> None:
        """Export results to a CSV file.

        Args:
            path: Destination file path.
        """
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Downstream results saved to %s", path)

    def to_json(self, path: str | Path) -> None:
        """Export results to a JSON file.

        Args:
            path: Destination file path.
        """
        Path(path).write_text(json.dumps(self.records, indent=2, default=_json_default))
        logger.info("Downstream results saved to %s", path)

    def summary(self) -> pd.DataFrame:
        """Return a summary table grouped by task.

        Pivots the results so each metric becomes a column, with one row per
        (task, task_type) combination.

        Returns:
            DataFrame with task, task_type as index columns and metric names
            as value columns.
        """
        df = self.to_dataframe()
        if df.empty:
            return df
        pivot = df.pivot_table(
            index=["task", "task_type"],
            columns="metric",
            values="value",
            aggfunc="first",
        ).reset_index()
        return pivot

    def to_submission_yaml(
        self,
        method_name: str,
        submitter_team: str,
        code_url: str,
        paper_url: str = "",
        track: str = "Track 1 — Health & Behavior Outcome Prediction (Static)",
        method_category: str = "Other",
        foundation_variant: str = "N/A (not a foundation model)",
        feature_dim: str = "—",
        notes: str = "",
    ) -> str:
        """Render a paste-ready leaderboard-submission body.

        See `.github/ISSUE_TEMPLATE/submission.yml` for field semantics.
        Skill scores are emitted as "—" placeholders for Track 1; maintainers
        fill them in during ingestion. ``paper_url`` is optional — leave
        empty for independent submissions without a write-up.
        """
        from openmhc._submission import prediction_to_submission_yaml

        return prediction_to_submission_yaml(
            self,
            method_name=method_name,
            submitter_team=submitter_team,
            code_url=code_url,
            paper_url=paper_url,
            track=track,
            method_category=method_category,
            foundation_variant=foundation_variant,
            feature_dim=feature_dim,
            notes=notes,
        )

    def __repr__(self) -> str:
        n = len(self.records)
        return f"PredictionResults({n} records, global_score={self.global_score:.4f})"


@dataclass
class ImputationResults:
    """Results from imputation evaluation.

    Attributes:
        scenarios: Dict mapping scenario name to per-split metric dicts.
            Structure: {scenario_name: {split_name: {group: {metric: value}}}}.
            Each per-split dict also carries ``overall_fallback_rate`` (scalar)
            and ``fallback_rate`` (per-channel) — the fraction of target cells
            the imputer left non-finite and that the harness substituted with
            a channel-aware global baseline. This is a **model-capability**
            metric, orthogonal to ``n_applicable``/``n_total`` (data quality):
            ``n_applicable`` reports samples a masking scenario could be
            applied to; ``overall_fallback_rate`` reports the fraction of
            target cells the model itself failed to produce.
    """

    scenarios: dict = field(repr=False)

    @property
    def overall_fallback_rate(self) -> float:
        """Max ``overall_fallback_rate`` across all (scenario, split) entries.

        Returns 0.0 when no scenario/split reports a fallback rate (e.g. the
        harness was run with no fallback fill, or every model output was
        finite at target cells). Mirrors ``ForecastingResults`` in surfacing
        the worst-case substitution rate at the top level.
        """
        worst = 0.0
        for split_map in self.scenarios.values():
            if not isinstance(split_map, dict):
                continue
            for metrics in split_map.values():
                if not isinstance(metrics, dict):
                    continue
                rate = metrics.get("overall_fallback_rate")
                if isinstance(rate, (int, float)) and rate > worst:
                    worst = float(rate)
        return worst

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten scenario results into a DataFrame.

        Returns:
            DataFrame with columns: scenario, split, channel_group, metric,
            value.
        """
        rows = []
        for scenario, splits in self.scenarios.items():
            for split_name, split_data in splits.items():
                if not isinstance(split_data, dict):
                    continue
                for group_name, group_metrics in split_data.items():
                    if not isinstance(group_metrics, dict):
                        # Top-level scalar like n_samples.
                        rows.append(
                            {
                                "scenario": scenario,
                                "split": split_name,
                                "channel_group": "_meta",
                                "metric": group_name,
                                "value": group_metrics,
                            }
                        )
                        continue
                    for metric_name, value in group_metrics.items():
                        rows.append(
                            {
                                "scenario": scenario,
                                "split": split_name,
                                "channel_group": group_name,
                                "metric": metric_name,
                                "value": value,
                            }
                        )
        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path) -> None:
        """Export results to a CSV file.

        Args:
            path: Destination file path.
        """
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Imputation results saved to %s", path)

    def to_json(self, path: str | Path) -> None:
        """Export results to a JSON file.

        Args:
            path: Destination file path.
        """
        Path(path).write_text(json.dumps(self.scenarios, indent=2, default=_json_default))
        logger.info("Imputation results saved to %s", path)

    def summary(self) -> pd.DataFrame:
        """Return a summary table with one row per (scenario, split).

        Filters to aggregate metric groups (continuous, binary) and pivots so
        each metric becomes a column.

        Returns:
            DataFrame with scenario, split, channel_group as index columns
            and metric names as value columns.
        """
        df = self.to_dataframe()
        if df.empty:
            return df
        agg = df[df["channel_group"].isin(["continuous", "binary"])]
        if agg.empty:
            return df
        pivot = agg.pivot_table(
            index=["scenario", "split", "channel_group"],
            columns="metric",
            values="value",
            aggfunc="first",
        ).reset_index()
        return pivot

    def to_submission_yaml(
        self,
        method_name: str,
        submitter_team: str,
        code_url: str,
        paper_url: str = "",
        method_category: str = "Other",
        foundation_variant: str = "N/A (not a foundation model)",
        feature_dim: str = "—",
        notes: str = "",
    ) -> str:
        """Render a paste-ready leaderboard-submission body.

        See `.github/ISSUE_TEMPLATE/submission.yml` for field semantics.
        Skill scores are computed locally from frozen LOCF baselines
        (`data/baselines/imputation_locf.json`) when the file is present,
        else emitted as "—". ``paper_url`` is optional — leave empty for
        independent submissions without a write-up.
        """
        from openmhc._submission import imputation_to_submission_yaml

        return imputation_to_submission_yaml(
            self,
            method_name=method_name,
            submitter_team=submitter_team,
            code_url=code_url,
            paper_url=paper_url,
            method_category=method_category,
            foundation_variant=foundation_variant,
            feature_dim=feature_dim,
            notes=notes,
        )

    def __repr__(self) -> str:
        n = len(self.scenarios)
        return f"ImputationResults({n} scenarios)"


@dataclass
class ForecastingResults:
    """Results from forecasting evaluation (Track 3).

    Attributes:
        per_channel: Dict mapping channel name (e.g. ``"hr"``) to a per-metric
            dict (``mae``, ``mase``, ``ql``, ``sql`` for continuous; ``auprc``,
            ``auroc`` for binary).
        run_dir: Path to the directory where per-user prediction parquets +
            offline metric outputs were written.
        n_samples: Total prediction samples emitted.
        overall_fallback_rate: Fraction of forecast cells where the model
            returned NaN and the Seasonal-Naive baseline was substituted before
            scoring. A high value means the model could not predict much of the
            in-scope window set; metrics should be read alongside this number.
        fallback_rate: Per-channel Seasonal-Naive substitution fractions, keyed
            like ``per_channel`` (``ch_<i>``).
    """

    per_channel: dict = field(repr=False)
    run_dir: str = ""
    n_samples: int = 0
    overall_fallback_rate: float = 0.0
    fallback_rate: dict = field(default_factory=dict, repr=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten per-channel metrics into a long DataFrame.

        Per-channel Seasonal-Naive ``fallback_rate`` is included as an extra
        metric row so it surfaces alongside the error metrics.
        """
        rows = []
        for channel, metrics in self.per_channel.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                rows.append({"channel": channel, "metric": metric_name, "value": value})
        for channel, rate in self.fallback_rate.items():
            rows.append({"channel": channel, "metric": "fallback_rate", "value": rate})
        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path) -> None:
        """Export per-channel results to CSV."""
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Forecasting results saved to %s", path)

    def to_json(self, path: str | Path) -> None:
        """Export full results dict to JSON."""
        Path(path).write_text(
            json.dumps(self.per_channel, indent=2, default=_json_default)
        )
        logger.info("Forecasting results saved to %s", path)

    def summary(self) -> pd.DataFrame:
        """Wide table: rows = channels, cols = metrics."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return df.pivot_table(
            index="channel", columns="metric", values="value", aggfunc="first"
        ).reset_index()

    def to_submission_yaml(
        self,
        method_name: str,
        submitter_team: str,
        code_url: str,
        paper_url: str = "",
        method_category: str = "Other",
        foundation_variant: str = "N/A (not a foundation model)",
        feature_dim: str = "—",
        notes: str = "",
    ) -> str:
        """Render a paste-ready submission body for a Track 3 forecasting result.

        Skill scores against the Seasonal Naive baseline are emitted as ``—``
        because the per-channel baseline file isn't shipped yet; maintainers
        fill them in from ``raw_metrics`` during ingestion.
        """
        from openmhc._submission import forecasting_to_submission_yaml

        return forecasting_to_submission_yaml(
            self,
            method_name=method_name,
            submitter_team=submitter_team,
            code_url=code_url,
            paper_url=paper_url,
            method_category=method_category,
            foundation_variant=foundation_variant,
            feature_dim=feature_dim,
            notes=notes,
        )

    def __repr__(self) -> str:
        n = len(self.per_channel)
        return (
            f"ForecastingResults({n} channels, n_samples={self.n_samples}, "
            f"overall_fallback_rate={self.overall_fallback_rate:.4f})"
        )


def _json_default(obj):
    """Serialize numpy types for JSON encoding.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-compatible Python scalar or list.

    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
