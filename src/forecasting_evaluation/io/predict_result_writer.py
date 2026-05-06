"""Parquet writers for forecasting prediction results."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from forecasting_evaluation.config import ForecastingEvalConfig

logger = logging.getLogger(__name__)


def _make_type_nullable(data_type: Any, pa_module: Any) -> Any:
    """Recursively rebuild pyarrow data types with nullable nested fields."""
    if pa_module.types.is_struct(data_type):
        return pa_module.struct(
            [
                pa_module.field(
                    field.name,
                    _make_type_nullable(field.type, pa_module),
                    nullable=True,
                    metadata=field.metadata,
                )
                for field in data_type
            ]
        )

    if pa_module.types.is_list(data_type):
        value_field = data_type.value_field
        return pa_module.list_(
            pa_module.field(
                value_field.name,
                _make_type_nullable(value_field.type, pa_module),
                nullable=True,
                metadata=value_field.metadata,
            )
        )

    if pa_module.types.is_large_list(data_type):
        value_field = data_type.value_field
        return pa_module.large_list(
            pa_module.field(
                value_field.name,
                _make_type_nullable(value_field.type, pa_module),
                nullable=True,
                metadata=value_field.metadata,
            )
        )

    if pa_module.types.is_fixed_size_list(data_type):
        value_field = data_type.value_field
        return pa_module.list_(
            pa_module.field(
                value_field.name,
                _make_type_nullable(value_field.type, pa_module),
                nullable=True,
                metadata=value_field.metadata,
            )
        )

    return data_type


def _make_schema_nullable(schema: Any, pa_module: Any) -> Any:
    """Ensure all schema fields (including nested list/struct fields) are nullable."""
    return pa_module.schema(
        [
            pa_module.field(
                field.name,
                _make_type_nullable(field.type, pa_module),
                nullable=True,
                metadata=field.metadata,
            )
            for field in schema
        ],
        metadata=schema.metadata,
    )


def to_serializable(value: Any) -> Any:
    """Convert nested numpy objects to native Python types for parquet compatibility.

    Handles non-finite floating values by normalizing them to NaN so nested
    numeric arrays remain valid float arrays when written through PyArrow.
    """
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        # Recursively process the list result to convert NaN/inf to None
        list_value = value.tolist()
        return to_serializable(list_value)
    if isinstance(value, np.generic):
        result = value.item()
        # Keep nested numeric arrays float-typed for stable pyarrow conversion.
        if isinstance(result, float) and not np.isfinite(result):
            return float("nan")
        return result
    # Handle native Python floats
    if isinstance(value, float):
        if not np.isfinite(value):
            return float("nan")
    return value


def sanitize_name(name: str) -> str:
    """Convert name into filesystem-safe string."""
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return sanitized or "unknown"


class PublicWriter:
    """Handle run-level metadata and shared output context."""

    def __init__(
        self,
        config: ForecastingEvalConfig,
        experiment_name: str | None = None,
    ) -> None:
        """Initialize run-level output context and metadata writer.

        Args:
            config: Forecasting evaluation config containing output settings.
            experiment_name: Optional run namespace used under results_dir.
        """
        self.config = config

        self.experiment_name = experiment_name or config.experiment_name or "Default"
        self.experiment_dir = Path(config.output.results_dir) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir: Path | None = None
        self.model_name: str | None = None

        self.total_written = 0
        self.skipped_users = 0

    def prepare_model_dir(self, model_name: str) -> Path:
        """Create and return current model output directory under the experiment directory."""
        model_dir_name = sanitize_name(model_name)
        if self.experiment_name.startswith("Test"):
            model_dir_name = f"{model_dir_name}_{datetime.now().strftime('%Y%m%d%H%M')}"

        self.model_name = model_name
        self.model_dir = self.experiment_dir / model_dir_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self.config.output.save_config:
            self._save_run_config()
        return self.model_dir

    def get_user_file_path(self, user_id: str) -> Path:
        """Get deterministic parquet file path for one user under current model directory."""
        if self.model_dir is None:
            raise RuntimeError("Model output directory is not initialized")
        return self.model_dir / f"{sanitize_name(user_id)}.parquet"

    def should_skip_user(self, user_id: str) -> bool:
        """Return whether existing user parquet should be skipped under overwrite policy."""
        if self.config.output.overwrite_existing_parquet:
            return False
        return self.get_user_file_path(user_id).exists()

    def increment_written(self, count: int = 1) -> None:
        """Increase global prediction row count."""
        self.total_written += count

    def increment_skipped_users(self, count: int = 1) -> None:
        """Increase skipped user count when existing outputs are preserved."""
        self.skipped_users += count

    def finalize(self) -> Path:
        """Finalize run-level output and return run directory."""
        logger.info("Prediction write complete. Total samples: %d", self.total_written)
        logger.info("Skipped users (existing parquet): %d", self.skipped_users)
        if self.model_dir is None:
            raise RuntimeError("Model output directory is not initialized")
        logger.info("Run directory: %s", self.model_dir)
        return self.model_dir

    def _save_run_config(self) -> None:
        """Persist run config for reproducibility."""
        try:
            if is_dataclass(self.config):
                config_data = asdict(self.config)
            else:
                config_data = dict(self.config)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to serialize run config: %s", exc)
            return

        if self.model_dir is None:
            raise RuntimeError("Model output directory is not initialized")

        yaml_path = self.model_dir / "config.yaml"
        json_path = self.model_dir / "config.json"

        try:
            import yaml

            with yaml_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config_data, handle, sort_keys=False, allow_unicode=True)
            return
        except ImportError:
            logger.warning("PyYAML not available, saving JSON config instead")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed writing YAML config, saving JSON config instead: %s", exc)

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(config_data, handle, indent=2, ensure_ascii=False)

class PredictResultWriter:
    """Write prediction rows for a single model/user pair."""

    def __init__(self, model_dir: Path, user_id: str, overwrite_existing: bool = False) -> None:
        """Initialize a parquet writer for one model-user output shard.

        Args:
            model_dir: Directory for one model output.
            user_id: User identifier used in parquet filename.
            overwrite_existing: Whether to overwrite existing user parquet.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to write parquet prediction files") from exc

        self._pa = pa
        self._pq = pq
        self.model_name = model_dir.name
        self.user_id = user_id
        self.records_written = 0

        user_key = sanitize_name(user_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = model_dir / f"{user_key}.parquet"

        if self.file_path.exists() and not overwrite_existing:
            raise FileExistsError(f"Prediction output already exists: {self.file_path}")

        if self.file_path.exists() and overwrite_existing:
            self.file_path.unlink()

        self._writer: Any | None = None
        self._schema: Any | None = None

    def append(self, record: dict[str, Any]) -> None:
        """Append one prediction row into the current parquet file."""
        serializable_record = to_serializable(record)

        if self._writer is None:
            sample_table = self._pa.Table.from_pylist([serializable_record])
            self._schema = _make_schema_nullable(sample_table.schema, self._pa)
            self._writer = self._pq.ParquetWriter(self.file_path, self._schema)

        table = self._pa.Table.from_pylist([serializable_record], schema=self._schema)
        self._writer.write_table(table)
        self.records_written += 1

    def close(self) -> None:
        """Close parquet writer for current model/user pair."""
        if self._writer is None:
            return

        self._writer.close()
        self._writer = None
        self._schema = None
        logger.debug("Finalized prediction file: %s (rows=%d)", self.file_path, self.records_written)
