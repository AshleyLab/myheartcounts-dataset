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
        """Render the Track 1 leaderboard submission packet.

        Returns the ``meta.json`` sidecar block plus the pull-request checklist
        for the Hugging Face dataset repo
        ``MyHeartCounts/OpenMHC-leaderboard-data``; the maintainers compute
        skill / fair-skill / rank from the substrate during ingestion.
        ``paper_url`` is optional — leave empty for independent submissions
        without a write-up.
        """
        from openmhc._submission import prediction_to_submission_yaml

        return prediction_to_submission_yaml(
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
        per_user_errors: Optional ``DataFrame`` of per-(user, channel, cell)
            errors emitted by the canonical producer
            (:func:`imputation_evaluation.evaluation.per_user_errors.build_per_user_errors`).
            Schema: ``[method, scenario, split, channel, channel_type,
            subgroup_attr, subgroup_value, user_id, E_per_user]`` — the
            same long format consumed by the paper Phase 2 aggregators
            and the BCa LOO jackknife. Populated by
            ``evaluate_imputation``; when ``output_dir`` is set this frame
            is also written to
            ``<output_dir>/per_user_errors.parquet``.
        skill_scores: Optional ``DataFrame`` of paired-R skill scores
            against the baseline supplied via ``baseline_errors``.
            Columns: ``[method, scope, skill_score, n_tasks]`` —
            per-(scenario, channel-bucket, overall) scopes as emitted by
            :func:`paper_metrics_core.compute_skill_scores`. ``None``
            unless ``baseline_errors`` was provided.
    """

    scenarios: dict = field(repr=False)
    # Optional additive fields populated by evaluate_imputation when available.
    per_user_errors: pd.DataFrame | None = field(default=None, repr=False)
    skill_scores: pd.DataFrame | None = field(default=None, repr=False)

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
        """Render the Track 2 leaderboard submission packet.

        Returns the ``meta.json`` sidecar block plus the pull-request checklist
        for the Hugging Face dataset repo
        ``MyHeartCounts/OpenMHC-leaderboard-data``. The per-user substrate
        parquet (``per_user_errors.parquet``, written when you pass
        ``output_dir=`` to ``evaluate_imputation``) is the second PR file; the
        maintainers compute skill / fair-skill / rank from it vs. LOCF.
        ``paper_url`` is optional.
        """
        from openmhc._submission import imputation_to_submission_yaml

        return imputation_to_submission_yaml(
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
        Path(path).write_text(json.dumps(self.per_channel, indent=2, default=_json_default))
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
        """Render the Track 3 leaderboard submission packet.

        Returns the ``meta.json`` sidecar block plus the pull-request checklist
        for the Hugging Face dataset repo
        ``MyHeartCounts/OpenMHC-leaderboard-data``; the maintainers compute
        skill / fair-skill / rank vs. the Seasonal Naive baseline from the
        substrate during ingestion. The Track 3 subdir name and substrate
        format are still being finalized (the packet flags this).
        """
        from openmhc._submission import forecasting_to_submission_yaml

        return forecasting_to_submission_yaml(
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
