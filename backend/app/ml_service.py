"""Machine Learning service for IGANN model training and prediction."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from igann import IGANN
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .data_processing import prepare_training_data

DATA_FILE_NAME = "bike.csv"
MAX_PREVIEW_ROWS = 500
MODEL_ARTIFACT_VERSION = "1.0"
MODEL_ARTIFACT_VERSION_WITH_EDITS = "1.1"
SUPPORTED_MODEL_ARTIFACT_VERSIONS = {
    MODEL_ARTIFACT_VERSION,
    MODEL_ARTIFACT_VERSION_WITH_EDITS,
}


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas values to JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class ImportedIGANNRuntime:
    """Lightweight predictor that reconstructs model output from exported shape functions."""

    def __init__(
        self,
        *,
        feature_names: List[str],
        shape_functions: List[Dict[str, Any]],
        base_prediction_offset: float,
        igann_params: Dict[str, Any],
        effective_boosting_rounds: int,
    ) -> None:
        self.feature_names = list(feature_names)
        self.base_prediction_offset = float(base_prediction_offset)
        self._params = dict(igann_params or {})
        self.boosting_rates = [0.0] * max(int(effective_boosting_rounds), 0)
        self._shape_by_feature: Dict[str, Dict[str, Any]] = {}

        for shape_function in shape_functions:
            feature_name = str(shape_function.get("feature_name", "")).strip()
            if not feature_name:
                continue
            self._shape_by_feature[feature_name] = shape_function

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self._params)

    @staticmethod
    def _predict_numeric(shape_function: Dict[str, Any], value: Any) -> float:
        x_values = np.asarray(shape_function.get("x_values", []), dtype=float)
        y_values = np.asarray(shape_function.get("y_values", []), dtype=float)
        if x_values.size == 0 or y_values.size == 0 or x_values.size != y_values.size:
            return 0.0
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return float(np.interp(numeric_value, x_values, y_values))

    @staticmethod
    def _predict_categorical(shape_function: Dict[str, Any], value: Any) -> float:
        x_values = [str(item) for item in shape_function.get("x_values", [])]
        y_values = [float(item) for item in shape_function.get("y_values", [])]
        if not x_values or len(x_values) != len(y_values):
            return 0.0
        value_str = str(value)
        try:
            idx = x_values.index(value_str)
        except ValueError:
            return 0.0
        return y_values[idx]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions: List[float] = []
        for _, row in X.iterrows():
            total = self.base_prediction_offset
            for feature_name in self.feature_names:
                shape_function = self._shape_by_feature.get(feature_name)
                if shape_function is None:
                    continue
                if shape_function.get("feature_type") == "categorical":
                    total += self._predict_categorical(shape_function, row.get(feature_name))
                else:
                    total += self._predict_numeric(shape_function, row.get(feature_name))
            predictions.append(float(total))
        return np.asarray(predictions, dtype=float)


def _candidate_data_files() -> List[Path]:
    """Return likely built-in dataset locations across runtime layouts."""
    app_dir = Path(__file__).resolve().parent
    backend_dir = app_dir.parent
    repo_root = backend_dir.parent
    cwd = Path.cwd().resolve()

    candidates: List[Path] = []
    configured = os.getenv("DATA_FILE_PATH", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            repo_root / DATA_FILE_NAME,
            backend_dir / DATA_FILE_NAME,
            cwd / DATA_FILE_NAME,
            cwd / "backend" / DATA_FILE_NAME,
        ]
    )

    seen = set()
    deduped: List[Path] = []
    for path in candidates:
        normalized = path.resolve() if not path.is_absolute() else path
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _resolve_fallback_data_file() -> Path:
    """Resolve the default dataset path and raise a clear error if missing."""
    checked_paths: List[Path] = []
    for candidate in _candidate_data_files():
        checked_paths.append(candidate)
        if candidate.is_file():
            return candidate

    checked = "\n".join(f"- {path}" for path in checked_paths)
    raise FileNotFoundError(
        "Dataset not found. Checked the following locations:\n"
        f"{checked}\n"
        "Set DATA_FILE_PATH to a valid CSV path if needed."
    )


class MLService:
    """Service for managing IGANN models, datasets and predictions."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.preprocessor = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

        self.is_trained = False
        self.feature_names: List[str] = []
        self.selected_feature_columns: List[str] = []
        self.cat_features: List[str] = []
        self.num_features: List[str] = []
        self.target_column: Optional[str] = None

        self.feature_schema: List[Dict[str, Any]] = []
        self.feature_schema_map: Dict[str, Dict[str, Any]] = {}
        self.model_source: str = "trained"
        self.imported_artifact_version: Optional[str] = None
        self.analytics_available = False
        self.primary_n_estimators: Optional[int] = None

        # Store original shape functions for interactive editing
        self.original_shape_functions: Dict[str, Dict[str, Any]] = {}
        self.shape_function_offsets: Dict[str, Dict[Any, float]] = {}

        # Comparison dataset/model runtime (superadmin-only analysis state)
        self.comparison_model: Optional[Any] = None
        self.comparison_preprocessor = None
        self.comparison_X_train: Optional[pd.DataFrame] = None
        self.comparison_X_test: Optional[pd.DataFrame] = None
        self.comparison_y_train: Optional[pd.DataFrame] = None
        self.comparison_y_test: Optional[pd.DataFrame] = None
        self.comparison_df: Optional[pd.DataFrame] = None
        self.comparison_is_trained = False
        self.comparison_feature_names: List[str] = []
        self.comparison_selected_feature_columns: List[str] = []
        self.comparison_cat_features: List[str] = []
        self.comparison_num_features: List[str] = []
        self.comparison_target_column: Optional[str] = None
        self.comparison_feature_schema: List[Dict[str, Any]] = []
        self.comparison_feature_schema_map: Dict[str, Dict[str, Any]] = {}
        self.comparison_original_shape_functions: Dict[str, Dict[str, Any]] = {}
        self.comparison_shape_function_offsets: Dict[str, Dict[Any, float]] = {}
        self.comparison_dataset_id: Optional[str] = None
        self.comparison_dataset_path: Optional[str] = None
        self.comparison_dataset_name: Optional[str] = None

        # Dataset persistence
        backend_dir = Path(__file__).resolve().parent.parent
        self.data_store_dir = backend_dir / "data_store"
        self.datasets_dir = self.data_store_dir / "datasets"
        self.active_dataset_file = self.data_store_dir / "active_dataset.json"
        self.data_store_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.active_dataset_id: Optional[str] = None
        self.active_dataset_path: Optional[str] = None
        self.active_dataset_name: Optional[str] = None
        self.feature_chart_settings: Dict[str, Dict[str, Any]] = {}

        self._restore_active_dataset_metadata()
        self._auto_load_persisted_dataset()

    def _clear_comparison_state(self) -> None:
        self.comparison_model = None
        self.comparison_preprocessor = None
        self.comparison_X_train = None
        self.comparison_X_test = None
        self.comparison_y_train = None
        self.comparison_y_test = None
        self.comparison_df = None
        self.comparison_is_trained = False
        self.comparison_feature_names = []
        self.comparison_selected_feature_columns = []
        self.comparison_cat_features = []
        self.comparison_num_features = []
        self.comparison_target_column = None
        self.comparison_feature_schema = []
        self.comparison_feature_schema_map = {}
        self.comparison_original_shape_functions = {}
        self.comparison_shape_function_offsets = {}
        self.comparison_dataset_id = None
        self.comparison_dataset_path = None
        self.comparison_dataset_name = None

    def _snapshot_runtime_state(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "df": self.df,
            "is_trained": self.is_trained,
            "feature_names": list(self.feature_names),
            "selected_feature_columns": list(self.selected_feature_columns),
            "cat_features": list(self.cat_features),
            "num_features": list(self.num_features),
            "target_column": self.target_column,
            "feature_schema": list(self.feature_schema),
            "feature_schema_map": dict(self.feature_schema_map),
            "original_shape_functions": dict(self.original_shape_functions),
            "shape_function_offsets": {
                feature: dict(offsets)
                for feature, offsets in self.shape_function_offsets.items()
            },
            "model_source": self.model_source,
            "imported_artifact_version": self.imported_artifact_version,
            "analytics_available": self.analytics_available,
        }

    def _restore_runtime_state(self, snapshot: Dict[str, Any]) -> None:
        for key, value in snapshot.items():
            setattr(self, key, value)

    # ==================== Dataset management ====================

    def _restore_active_dataset_metadata(self) -> None:
        if not self.active_dataset_file.exists():
            return
        try:
            data = json.loads(self.active_dataset_file.read_text(encoding="utf-8"))
        except Exception:
            return

        dataset_path = data.get("dataset_path")
        target_column = data.get("target_column")
        selected_feature_columns = data.get("selected_feature_columns")
        dataset_id = data.get("dataset_id")
        dataset_name = data.get("dataset_name")
        feature_chart_settings = data.get("feature_chart_settings")
        if dataset_path:
            self.active_dataset_path = str(Path(dataset_path))
        if target_column:
            self.target_column = str(target_column)
        if isinstance(selected_feature_columns, list):
            self.selected_feature_columns = [str(col) for col in selected_feature_columns]
        if dataset_id:
            self.active_dataset_id = str(dataset_id)
        if dataset_name:
            self.active_dataset_name = str(Path(str(dataset_name)).name)
        if isinstance(feature_chart_settings, dict):
            sanitized: Dict[str, Dict[str, Any]] = {}
            for feature_name, raw_setting in feature_chart_settings.items():
                if not isinstance(raw_setting, dict):
                    continue
                raw_labels = raw_setting.get("value_labels")
                labels: Dict[str, str] = {}
                if isinstance(raw_labels, dict):
                    for raw_key, raw_label in raw_labels.items():
                        label_str = str(raw_label).strip()
                        if not label_str:
                            continue
                        labels[str(raw_key)] = label_str
                sanitized[str(feature_name)] = {
                    "treat_as_categorical": bool(raw_setting.get("treat_as_categorical")),
                    "treat_as_numeric": bool(raw_setting.get("treat_as_numeric")),
                    "value_labels": labels,
                }
            self.feature_chart_settings = sanitized

    def _persist_active_dataset_metadata(self) -> None:
        payload = {
            "dataset_id": self.active_dataset_id,
            "dataset_path": self.active_dataset_path,
            "dataset_name": self.active_dataset_name,
            "target_column": self.target_column,
            "selected_feature_columns": self.selected_feature_columns,
            "feature_chart_settings": self.feature_chart_settings,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.active_dataset_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _auto_load_persisted_dataset(self) -> None:
        if not self.active_dataset_path or not self.target_column:
            return
        dataset_path = Path(self.active_dataset_path)
        if not dataset_path.is_file():
            return
        try:
            self.load_data(
                dataset_id=self.active_dataset_id,
                target_column=self.target_column,
                feature_columns=self.selected_feature_columns or None,
            )
        except Exception:
            # Keep service usable even if persisted dataset is invalid.
            self._reset_runtime_state()

    def _resolve_dataset_path(self, dataset_id: Optional[str] = None) -> Tuple[Path, Optional[str]]:
        if dataset_id:
            safe_name = Path(dataset_id).name
            candidate = (self.datasets_dir / safe_name).resolve()
            if not str(candidate).startswith(str(self.datasets_dir.resolve())):
                raise ValueError("Invalid dataset_id")
            if not candidate.is_file():
                raise FileNotFoundError(f"Uploaded dataset not found: {safe_name}")
            return candidate, safe_name

        if self.active_dataset_path:
            active_path = Path(self.active_dataset_path).resolve()
            if active_path.is_file():
                return active_path, self.active_dataset_id

        return _resolve_fallback_data_file(), None

    def inspect_uploaded_dataset(self, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        """Persist an uploaded CSV and return column preview information."""
        if not file_name or not file_name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported")
        if not file_bytes:
            raise ValueError("Uploaded file is empty")

        dataset_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex}.csv"
        dataset_path = self.datasets_dir / dataset_id
        dataset_path.write_bytes(file_bytes)

        try:
            preview_df = pd.read_csv(dataset_path, nrows=MAX_PREVIEW_ROWS)
        except Exception as exc:
            dataset_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to parse CSV: {exc}") from exc

        columns = [str(col) for col in preview_df.columns]
        if len(columns) < 2:
            dataset_path.unlink(missing_ok=True)
            raise ValueError("CSV must contain at least one target and one feature column")

        default_target = "cnt" if "cnt" in columns else columns[-1]
        return {
            "dataset_id": dataset_id,
            "original_filename": file_name,
            "columns": columns,
            "default_target_column": default_target,
        }

    @staticmethod
    def _stringify_chart_value(value: Any) -> str:
        """Convert raw values into stable, human-readable chart keys."""
        if value is None:
            return "Unknown"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                return str(numeric)
            rounded = round(numeric)
            if abs(numeric - rounded) < 1e-9:
                return str(int(rounded))
            return format(numeric, ".12g")
        return str(value)

    @staticmethod
    def _is_integer_like_numeric(values: pd.Series) -> bool:
        if values.empty:
            return False
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return False
        arr = numeric.to_numpy(dtype=float)
        return bool(np.all(np.isclose(arr, np.round(arr))))

    def _feature_can_be_categorical(self, feature_name: str) -> bool:
        """Whether a feature can safely be treated as categorical for charts."""
        if feature_name in self.cat_features:
            return True
        if self.X_train is None or feature_name not in self.X_train.columns:
            return False

        series = pd.to_numeric(self.X_train[feature_name], errors="coerce").dropna()
        if series.empty or not self._is_integer_like_numeric(series):
            return False

        unique_count = int(series.nunique(dropna=True))
        return unique_count <= 13

    def _feature_can_be_numeric(self, feature_name: str) -> bool:
        """Whether a feature can safely be treated as numeric for charts."""
        if feature_name in self.num_features:
            return True
        if self.X_train is None or feature_name not in self.X_train.columns:
            return False

        series = self.X_train[feature_name]
        non_null = series.dropna()
        if non_null.empty:
            return False

        numeric = pd.to_numeric(non_null, errors="coerce")
        return bool(numeric.notna().all())

    def _get_feature_chart_setting(self, feature_name: str) -> Dict[str, Any]:
        raw = self.feature_chart_settings.get(feature_name, {})
        labels = raw.get("value_labels", {}) if isinstance(raw, dict) else {}
        sanitized_labels = {
            str(k): str(v).strip()
            for k, v in labels.items()
            if str(v).strip()
        } if isinstance(labels, dict) else {}
        return {
            "treat_as_categorical": bool(raw.get("treat_as_categorical")) if isinstance(raw, dict) else False,
            "treat_as_numeric": bool(raw.get("treat_as_numeric")) if isinstance(raw, dict) else False,
            "value_labels": sanitized_labels,
        }

    def _get_feature_chart_mode(self, feature_name: str) -> str:
        """Return effective chart display mode ('numeric' or 'categorical')."""
        setting = self._get_feature_chart_setting(feature_name)
        base_feature_type = "categorical" if feature_name in self.cat_features else "numeric"

        if base_feature_type == "numeric":
            if bool(setting.get("treat_as_categorical")) and self._feature_can_be_categorical(feature_name):
                return "categorical"
            return "numeric"

        if bool(setting.get("treat_as_numeric")) and self._feature_can_be_numeric(feature_name):
            return "numeric"
        return "categorical"

    def _is_feature_categorical_for_chart(self, feature_name: str) -> bool:
        return self._get_feature_chart_mode(feature_name) == "categorical"

    def _get_chart_x_values(
        self,
        feature_name: str,
        *,
        treat_as_categorical: Optional[bool] = None,
    ) -> List[str]:
        """Get raw x-axis values that should be used for a categorical chart."""
        if self.X_train is None or feature_name not in self.X_train.columns:
            return []

        if treat_as_categorical is None:
            treat_as_categorical = self._is_feature_categorical_for_chart(feature_name)

        if not treat_as_categorical:
            return []

        if feature_name in self.cat_features:
            options = self.feature_schema_map.get(feature_name, {}).get("categorical_options", [])
            base_values = [self._stringify_chart_value(v) for v in options]
        else:
            numeric_series = pd.to_numeric(self.X_train[feature_name], errors="coerce").dropna()
            uniques = sorted(numeric_series.unique().tolist())
            base_values = [self._stringify_chart_value(v) for v in uniques]

        deduped: List[str] = []
        seen = set()
        for raw in base_values:
            if raw in seen:
                continue
            seen.add(raw)
            deduped.append(raw)

        # Keep intuitive ordering for integer-coded categories like month (1..12).
        numeric_pairs: List[Tuple[float, str]] = []
        for raw in deduped:
            try:
                numeric_pairs.append((float(raw), raw))
            except (TypeError, ValueError):
                numeric_pairs = []
                break
        if numeric_pairs:
            numeric_pairs.sort(key=lambda item: item[0])
            deduped = [raw for _, raw in numeric_pairs]
        return deduped

    def _get_numeric_chart_values_for_categorical_feature(
        self,
        feature_name: str,
    ) -> List[Tuple[str, float]]:
        """Return (raw_value, numeric_x) pairs for categorical-as-numeric display."""
        raw_values = self._get_chart_x_values(
            feature_name,
            treat_as_categorical=True,
        )

        pairs: List[Tuple[str, float]] = []
        for raw in raw_values:
            try:
                numeric_val = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_val):
                continue
            pairs.append((str(raw), float(numeric_val)))

        pairs.sort(key=lambda item: item[1])
        return pairs

    def get_feature_chart_setting(self, feature_name: str) -> Dict[str, Any]:
        """Return effective chart settings for one feature."""
        if feature_name not in self.feature_names:
            raise ValueError(f"Unknown feature: {feature_name}")

        base_feature_type = "categorical" if feature_name in self.cat_features else "numeric"
        can_be_categorical = self._feature_can_be_categorical(feature_name)
        can_be_numeric = self._feature_can_be_numeric(feature_name)
        stored = self._get_feature_chart_setting(feature_name)
        treat_as_categorical = bool(stored.get("treat_as_categorical")) if base_feature_type == "numeric" else False
        treat_as_numeric = bool(stored.get("treat_as_numeric")) if base_feature_type == "categorical" else False

        if treat_as_categorical and not can_be_categorical:
            treat_as_categorical = False
        if treat_as_numeric and not can_be_numeric:
            treat_as_numeric = False

        chart_feature_type = "categorical"
        if base_feature_type == "numeric":
            chart_feature_type = "categorical" if treat_as_categorical else "numeric"
        else:
            chart_feature_type = "numeric" if treat_as_numeric else "categorical"

        is_categorical = chart_feature_type == "categorical"
        is_numeric = chart_feature_type == "numeric"
        allowed_values = set(
            self._get_chart_x_values(
                feature_name,
                treat_as_categorical=is_categorical,
            )
        )
        available_values = self._get_chart_x_values(
            feature_name,
            treat_as_categorical=bool(
                is_categorical
                or can_be_categorical
                or base_feature_type == "categorical"
            ),
        )
        value_labels = {
            str(raw): str(label).strip()
            for raw, label in (stored.get("value_labels", {}) or {}).items()
            if str(label).strip() and (not allowed_values or str(raw) in allowed_values)
        }

        return {
            "feature_name": feature_name,
            "base_feature_type": base_feature_type,
            "chart_feature_type": chart_feature_type,
            "is_numeric": is_numeric,
            "is_categorical": is_categorical,
            "can_be_categorical": can_be_categorical,
            "can_be_numeric": can_be_numeric,
            "treat_as_categorical": treat_as_categorical,
            "treat_as_numeric": treat_as_numeric,
            "value_labels": value_labels,
            "available_values": available_values,
        }

    def update_feature_chart_setting(
        self,
        feature_name: str,
        treat_as_categorical: bool,
        treat_as_numeric: bool = False,
        value_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Persist superadmin chart settings for one feature."""
        if self.X_train is None:
            raise ValueError("Data not loaded yet")
        if feature_name not in self.feature_names:
            raise ValueError(f"Unknown feature: {feature_name}")

        base_feature_type = "categorical" if feature_name in self.cat_features else "numeric"
        can_be_categorical = self._feature_can_be_categorical(feature_name)
        can_be_numeric = self._feature_can_be_numeric(feature_name)
        current = self._get_feature_chart_setting(feature_name)

        normalized_treat_as_categorical = False
        normalized_treat_as_numeric = False
        if base_feature_type == "numeric":
            normalized_treat_as_categorical = bool(treat_as_categorical)
            if normalized_treat_as_categorical and not can_be_categorical:
                raise ValueError(
                    f"Feature '{feature_name}' cannot be treated as categorical (too many distinct values)"
                )
        else:
            normalized_treat_as_numeric = bool(treat_as_numeric)
            if normalized_treat_as_numeric and not can_be_numeric:
                raise ValueError(
                    f"Feature '{feature_name}' cannot be treated as numeric (values are not numeric)"
                )

        effective_categorical = (
            base_feature_type == "categorical" and not normalized_treat_as_numeric
        ) or (
            base_feature_type == "numeric" and normalized_treat_as_categorical
        )
        allowed_values = set(
            self._get_chart_x_values(
                feature_name,
                treat_as_categorical=effective_categorical,
            )
        )

        source_labels = current.get("value_labels", {}) if value_labels is None else (value_labels or {})
        sanitized_labels: Dict[str, str] = {}
        for raw_key, raw_label in source_labels.items():
            key = self._stringify_chart_value(raw_key)
            label = str(raw_label).strip()
            if not label:
                continue
            if allowed_values and key not in allowed_values:
                continue
            sanitized_labels[key] = label

        normalized_setting = {
            "treat_as_categorical": normalized_treat_as_categorical,
            "treat_as_numeric": normalized_treat_as_numeric,
            "value_labels": sanitized_labels,
        }

        if (
            not normalized_setting["treat_as_categorical"]
            and not normalized_setting["treat_as_numeric"]
            and not normalized_setting["value_labels"]
        ):
            self.feature_chart_settings.pop(feature_name, None)
        else:
            self.feature_chart_settings[feature_name] = normalized_setting

        # Clear runtime chart snapshots/offsets so the next fetch reflects the new mode.
        self.original_shape_functions = {}
        self.shape_function_offsets = {}
        self._persist_active_dataset_metadata()
        return self.get_feature_chart_setting(feature_name)

    def _build_feature_schema_for(
        self,
        feature_names: List[str],
        num_features: List[str],
        X_all: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        schema: List[Dict[str, Any]] = []
        for feature_name in feature_names:
            if feature_name in num_features:
                numeric_series = pd.to_numeric(X_all[feature_name], errors="coerce")
                valid_numeric_series = numeric_series.dropna()

                if valid_numeric_series.empty:
                    min_val = 0.0
                    max_val = 0.0
                    default_val = 0.0
                else:
                    min_val = float(valid_numeric_series.min())
                    max_val = float(valid_numeric_series.max())
                    default_val = float(valid_numeric_series.median())

                    if default_val < min_val:
                        default_val = min_val
                    elif default_val > max_val:
                        default_val = max_val

                default_val = round(default_val, 2)

                schema.append(
                    {
                        "name": feature_name,
                        "feature_type": "numeric",
                        "default_value": default_val,
                        "min_value": min_val,
                        "max_value": max_val,
                    }
                )
            else:
                cat_series = X_all[feature_name].astype(str)
                unique_values = [str(v) for v in pd.unique(cat_series)]
                unique_values = sorted(unique_values)
                if not unique_values:
                    unique_values = ["Unknown"]
                value_counts = cat_series.value_counts()
                default_val = str(value_counts.index[0]) if not value_counts.empty else unique_values[0]
                if default_val not in unique_values:
                    default_val = unique_values[0]

                schema.append(
                    {
                        "name": feature_name,
                        "feature_type": "categorical",
                        "default_value": default_val,
                        "categorical_options": unique_values,
                    }
                )

        return schema

    def _build_feature_schema(self, X_all: pd.DataFrame) -> List[Dict[str, Any]]:
        return self._build_feature_schema_for(self.feature_names, self.num_features, X_all)

    def _has_analytics_data(self) -> bool:
        return (
            self.is_trained
            and self.X_test is not None
            and self.y_test is not None
            and self.model is not None
        )

    def _shape_functions_available(self) -> bool:
        return self.is_trained and (
            bool(self.original_shape_functions) or self.model is not None
        )

    def _require_analytics_data(self) -> None:
        if self._has_analytics_data():
            return
        if self.model_source == "imported":
            raise ValueError(
                "Analytics are unavailable for imported models until a compatible dataset is loaded"
            )
        raise ValueError("Model analytics are not available yet")

    def _ensure_original_shape_functions(self) -> None:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        if self.original_shape_functions:
            return
        if self.model_source == "imported":
            raise ValueError("Imported model shape functions are unavailable")

        existing_offsets = {
            feature: dict(offsets)
            for feature, offsets in self.shape_function_offsets.items()
        }
        shape_functions = self.get_shape_functions()
        self.original_shape_functions = {
            shape_function["feature_name"]: shape_function
            for shape_function in shape_functions
        }
        self.shape_function_offsets = existing_offsets

    def _export_shape_functions_payload(self) -> List[Dict[str, Any]]:
        if self.model_source == "imported":
            self._ensure_original_shape_functions()
            return [
                _json_safe(shape_function)
                for shape_function in self.original_shape_functions.values()
            ]

        if self.model is None or not hasattr(self.model, "get_shape_functions_as_dict"):
            raise ValueError("Unable to export complete shape functions for the trained model")

        raw_shape_functions = self.model.get_shape_functions_as_dict()
        exported: List[Dict[str, Any]] = []
        for feature_name in self.feature_names:
            raw_shape = raw_shape_functions.get(feature_name)
            if raw_shape is None:
                continue
            chart_config = {}
            if feature_name in self.feature_names:
                try:
                    chart_config = self.get_feature_chart_setting(feature_name)
                except Exception:
                    chart_config = {}
            feature_type = (
                "categorical"
                if str(raw_shape.get("datatype", "")).strip() == "categorical"
                else "numeric"
            )
            exported_shape = {
                "feature_name": feature_name,
                "x_values": _json_safe(raw_shape.get("x", [])),
                "y_values": _json_safe(raw_shape.get("y", [])),
                "feature_type": feature_type,
                "chart_config": chart_config,
            }
            if feature_type == "categorical":
                exported_shape["x_tick_labels"] = [
                    str(value) for value in _json_safe(raw_shape.get("x", []))
                ]
            exported.append(exported_shape)
        if not exported:
            raise ValueError("Unable to export any shape functions for the trained model")
        return exported

    def _infer_base_prediction_offset(self) -> float:
        if self.model is None:
            raise ValueError("Model not trained yet")
        if self.model_source == "imported":
            return float(getattr(self.model, "base_prediction_offset", 0.0))

        y_scaler = getattr(self.model, "y_scaler", None)
        mean_values = getattr(y_scaler, "mean_", None)
        if mean_values is not None and len(mean_values):
            return float(mean_values[0])
        intercept = getattr(getattr(self.model, "linear_model", None), "intercept_", 0.0)
        if isinstance(intercept, np.ndarray):
            if intercept.size:
                return float(intercept.reshape(-1)[0])
            return 0.0
        return float(intercept)

    def export_model_artifact(
        self,
        *,
        include_shape_function_edits: bool = False,
    ) -> Dict[str, Any]:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        igann_params = {}
        if hasattr(self.model, "get_params"):
            igann_params = _json_safe(self.model.get_params())

        shape_functions = self._export_shape_functions_payload()
        exported_feature_names = [
            shape_function["feature_name"] for shape_function in shape_functions
        ]
        exported_feature_schema = [
            _json_safe(item)
            for item in self.feature_schema
            if str(item.get("name")) in exported_feature_names
        ]
        exported_cat_features = [
            feature_name for feature_name in self.cat_features if feature_name in exported_feature_names
        ]
        exported_num_features = [
            feature_name for feature_name in self.num_features if feature_name in exported_feature_names
        ]

        artifact = {
            "artifact_version": (
                MODEL_ARTIFACT_VERSION_WITH_EDITS
                if include_shape_function_edits
                else MODEL_ARTIFACT_VERSION
            ),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "IGANN",
            "model_source": self.model_source,
            "igann_params": igann_params,
            "effective_boosting_rounds": int(len(getattr(self.model, "boosting_rates", []) or [])),
            "base_prediction_offset": float(self._infer_base_prediction_offset()),
            "feature_names": exported_feature_names,
            "selected_feature_columns": list(self.selected_feature_columns),
            "target_column": self.target_column,
            "feature_schema": exported_feature_schema,
            "cat_features": exported_cat_features,
            "num_features": exported_num_features,
            "feature_chart_settings": _json_safe(self.feature_chart_settings),
            "shape_functions": shape_functions,
            "dataset_id": self.active_dataset_id,
            "dataset_name": self.active_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None),
        }
        if include_shape_function_edits:
            artifact["shape_function_edits_export"] = {}
        return artifact

    @staticmethod
    def _validate_shape_function_artifact(shape_function: Dict[str, Any]) -> Dict[str, Any]:
        feature_name = str(shape_function.get("feature_name", "")).strip()
        if not feature_name:
            raise ValueError("Imported artifact contains a shape function without feature_name")

        feature_type = str(shape_function.get("feature_type", "")).strip()
        if feature_type not in {"numeric", "categorical"}:
            raise ValueError(f"Invalid feature_type for '{feature_name}'")

        x_values = shape_function.get("x_values")
        y_values = shape_function.get("y_values")
        if not isinstance(x_values, list) or not isinstance(y_values, list):
            raise ValueError(f"Shape function '{feature_name}' must include x_values and y_values arrays")
        if len(x_values) != len(y_values) or not x_values:
            raise ValueError(f"Shape function '{feature_name}' must have equally sized non-empty x/y arrays")

        normalized = {
            "feature_name": feature_name,
            "x_values": [str(x) for x in x_values] if feature_type == "categorical" else [float(x) for x in x_values],
            "y_values": [float(y) for y in y_values],
            "feature_type": feature_type,
            "chart_config": shape_function.get("chart_config") or {},
        }
        x_tick_labels = shape_function.get("x_tick_labels")
        if isinstance(x_tick_labels, list):
            normalized["x_tick_labels"] = [str(label) for label in x_tick_labels]
        return normalized

    @staticmethod
    def _validate_shape_function_edits_export(
        payload: Dict[str, Any],
        *,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Validate optional exported per-user shape-function edits."""
        included = bool(payload.get("included"))
        users_payload = payload.get("users", [])
        edits_payload = payload.get("edits", [])

        if not isinstance(users_payload, list) or not isinstance(edits_payload, list):
            raise ValueError(
                "Imported artifact shape_function_edits_export must include users and edits arrays"
            )

        normalized_users = []
        declared_user_names = set()
        for user in users_payload:
            if not isinstance(user, dict):
                raise ValueError("Imported artifact shape_function_edits_export users must be objects")
            user_name = str(user.get("name", "")).strip()
            if not user_name:
                raise ValueError("Imported artifact shape_function_edits_export contains a user without a name")
            if user_name in declared_user_names:
                raise ValueError(
                    f"Imported artifact shape_function_edits_export contains duplicate user '{user_name}'"
                )
            declared_user_names.add(user_name)
            normalized_users.append(
                {
                    "name": user_name,
                    "is_superadmin": False,
                }
            )

        normalized_edits = []
        submission_count = 0
        for user_group in edits_payload:
            if not isinstance(user_group, dict):
                raise ValueError(
                    "Imported artifact shape_function_edits_export edits must group entries by user"
                )
            user_name = str(user_group.get("user_name", "")).strip()
            if not user_name:
                raise ValueError(
                    "Imported artifact shape_function_edits_export contains an edit group without user_name"
                )
            if declared_user_names and user_name not in declared_user_names:
                raise ValueError(
                    f"Imported artifact shape_function_edits_export references undeclared user '{user_name}'"
                )

            shape_functions = user_group.get("shape_functions", [])
            if not isinstance(shape_functions, list):
                raise ValueError(
                    f"Imported artifact shape_function_edits_export for '{user_name}' must include shape_functions"
                )

            normalized_shape_functions = []
            for shape_function in shape_functions:
                if not isinstance(shape_function, dict):
                    raise ValueError(
                        "Imported artifact shape_function_edits_export shape function entries must be objects"
                    )
                feature_name = str(shape_function.get("feature_name", "")).strip()
                feature_type = str(shape_function.get("feature_type", "")).strip()
                if feature_name not in feature_names:
                    raise ValueError(
                        f"Imported artifact shape_function_edits_export references unknown feature '{feature_name}'"
                    )
                if feature_type not in {"numeric", "categorical"}:
                    raise ValueError(
                        f"Imported artifact shape_function_edits_export has invalid feature_type for '{feature_name}'"
                    )

                edited_points = shape_function.get("edited_points", [])
                if not isinstance(edited_points, list):
                    raise ValueError(
                        f"Imported artifact shape_function_edits_export for '{feature_name}' must include edited_points"
                    )

                normalized_points = []
                for point in edited_points:
                    if not isinstance(point, dict):
                        raise ValueError(
                            f"Imported artifact shape_function_edits_export points for '{feature_name}' must be objects"
                        )
                    normalized_point = {
                        "x_value": point.get("x_value"),
                        "y_value": float(point.get("y_value")),
                        "weight": float(point.get("weight", 0.5)),
                        "message": str(point.get("message", "") or ""),
                    }
                    if point.get("created_at"):
                        normalized_point["created_at"] = str(point.get("created_at"))
                    if point.get("updated_at"):
                        normalized_point["updated_at"] = str(point.get("updated_at"))
                    normalized_points.append(normalized_point)

                normalized_shape = {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "submission_id": str(shape_function.get("submission_id", "") or ""),
                    "message": str(shape_function.get("message", "") or ""),
                    "edited_points": normalized_points,
                }
                if shape_function.get("created_at"):
                    normalized_shape["created_at"] = str(shape_function.get("created_at"))
                if shape_function.get("updated_at"):
                    normalized_shape["updated_at"] = str(shape_function.get("updated_at"))
                if shape_function.get("sureness") is not None:
                    normalized_shape["sureness"] = int(shape_function.get("sureness"))
                normalized_shape_functions.append(normalized_shape)
                submission_count += 1

            normalized_edits.append(
                {
                    "user_name": user_name,
                    "shape_functions": normalized_shape_functions,
                }
            )

        return {
            "included": included,
            "users": normalized_users,
            "edits": normalized_edits,
            "user_count": len(normalized_users),
            "submission_count": submission_count,
        }

    def _validate_import_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = [
            "artifact_version",
            "model_type",
            "igann_params",
            "effective_boosting_rounds",
            "base_prediction_offset",
            "feature_names",
            "selected_feature_columns",
            "target_column",
            "feature_schema",
            "cat_features",
            "num_features",
            "feature_chart_settings",
            "shape_functions",
        ]
        missing = [key for key in required_keys if key not in artifact]
        if missing:
            raise ValueError(f"Imported artifact is missing required keys: {missing}")

        artifact_version = str(artifact.get("artifact_version", "")).strip()
        if artifact_version not in SUPPORTED_MODEL_ARTIFACT_VERSIONS:
            raise ValueError(
                "Unsupported artifact version "
                f"'{artifact_version}'. Expected one of {sorted(SUPPORTED_MODEL_ARTIFACT_VERSIONS)}"
            )
        if str(artifact.get("model_type", "")).strip() != "IGANN":
            raise ValueError("Only IGANN model artifacts are supported")

        feature_names = artifact.get("feature_names")
        selected_feature_columns = artifact.get("selected_feature_columns")
        feature_schema = artifact.get("feature_schema")
        cat_features = artifact.get("cat_features")
        num_features = artifact.get("num_features")
        shape_functions = artifact.get("shape_functions")

        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("Imported artifact must include non-empty feature_names")
        if not isinstance(selected_feature_columns, list) or not selected_feature_columns:
            raise ValueError("Imported artifact must include non-empty selected_feature_columns")
        if not isinstance(feature_schema, list) or len(feature_schema) != len(feature_names):
            raise ValueError("Imported artifact feature_schema must match the feature_names length")
        if not isinstance(cat_features, list) or not isinstance(num_features, list):
            raise ValueError("Imported artifact must include cat_features and num_features arrays")
        if sorted([str(item) for item in cat_features] + [str(item) for item in num_features]) != sorted([str(item) for item in feature_names]):
            raise ValueError("Imported artifact cat_features and num_features must cover all feature_names exactly")
        if not isinstance(shape_functions, list) or len(shape_functions) != len(feature_names):
            raise ValueError("Imported artifact must include one shape function per feature")

        feature_schema_map = {}
        for item in feature_schema:
            if not isinstance(item, dict):
                raise ValueError("Imported artifact feature_schema entries must be objects")
            feature_name = str(item.get("name", "")).strip()
            feature_type = str(item.get("feature_type", "")).strip()
            if feature_name not in feature_names:
                raise ValueError(f"Imported artifact schema references unknown feature '{feature_name}'")
            if feature_type not in {"numeric", "categorical"}:
                raise ValueError(f"Imported artifact schema has invalid feature_type for '{feature_name}'")
            feature_schema_map[feature_name] = _json_safe(item)

        if sorted(feature_schema_map.keys()) != sorted([str(item) for item in feature_names]):
            raise ValueError("Imported artifact feature_schema must cover all feature_names exactly")
        selected_feature_names = [str(item) for item in selected_feature_columns]
        if not set(feature_names).issubset(set(selected_feature_names)):
            raise ValueError("Imported artifact feature_names must be a subset of selected_feature_columns")

        normalized_shape_functions = [
            self._validate_shape_function_artifact(item)
            for item in shape_functions
        ]
        if sorted(sf["feature_name"] for sf in normalized_shape_functions) != sorted([str(item) for item in feature_names]):
            raise ValueError("Imported artifact shape_functions must cover all feature_names exactly")

        shape_function_edits_export = artifact.get("shape_function_edits_export")
        if shape_function_edits_export is not None and not isinstance(
            shape_function_edits_export,
            dict,
        ):
            raise ValueError(
                "Imported artifact shape_function_edits_export must be an object when present"
            )

        normalized_shape_function_edits_export = self._validate_shape_function_edits_export(
            _json_safe(shape_function_edits_export or {}),
            feature_names=[str(item) for item in feature_names],
        ) if shape_function_edits_export is not None else {}

        return {
            "artifact_version": artifact_version,
            "igann_params": dict(artifact.get("igann_params") or {}),
            "effective_boosting_rounds": int(artifact.get("effective_boosting_rounds") or 0),
            "base_prediction_offset": float(artifact.get("base_prediction_offset")),
            "feature_names": [str(item) for item in feature_names],
            "selected_feature_columns": selected_feature_names,
            "target_column": str(artifact.get("target_column")),
            "feature_schema": list(feature_schema_map.values()),
            "feature_schema_map": feature_schema_map,
            "cat_features": [str(item) for item in cat_features],
            "num_features": [str(item) for item in num_features],
            "feature_chart_settings": _json_safe(artifact.get("feature_chart_settings") or {}),
            "shape_functions": normalized_shape_functions,
            "dataset_id": artifact.get("dataset_id"),
            "dataset_name": artifact.get("dataset_name"),
            "shape_function_edits_export": normalized_shape_function_edits_export,
        }

    def import_model_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        validated = self._validate_import_artifact(artifact)
        imported_model = ImportedIGANNRuntime(
            feature_names=validated["feature_names"],
            shape_functions=validated["shape_functions"],
            base_prediction_offset=validated["base_prediction_offset"],
            igann_params=validated["igann_params"],
            effective_boosting_rounds=validated["effective_boosting_rounds"],
        )

        self.model = imported_model
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.is_trained = True
        self.feature_names = list(validated["feature_names"])
        self.selected_feature_columns = list(validated["selected_feature_columns"])
        self.cat_features = list(validated["cat_features"])
        self.num_features = list(validated["num_features"])
        self.target_column = validated["target_column"]
        self.feature_schema = list(validated["feature_schema"])
        self.feature_schema_map = dict(validated["feature_schema_map"])
        self.feature_chart_settings = dict(validated["feature_chart_settings"])
        self.original_shape_functions = {
            shape_function["feature_name"]: shape_function
            for shape_function in validated["shape_functions"]
        }
        self.shape_function_offsets = {}
        self.model_source = "imported"
        self.imported_artifact_version = validated["artifact_version"]
        self.analytics_available = False
        self.primary_n_estimators = None
        self.active_dataset_id = None
        self.active_dataset_path = None
        self.active_dataset_name = None
        self._clear_comparison_state()
        self.active_dataset_file.unlink(missing_ok=True)

        return {
            "success": True,
            "message": "Model imported successfully",
            "model_source": self.model_source,
            "imported_artifact_version": self.imported_artifact_version,
            "imported_shape_function_edits": bool(
                validated.get("shape_function_edits_export", {}).get("included")
            ),
            "imported_edit_user_count": 0,
            "imported_edit_submission_count": 0,
            "_shape_function_edits_export": validated.get(
                "shape_function_edits_export", {}
            ),
        }

    def _validate_dataset_compatibility(
        self,
        *,
        feature_names: List[str],
        feature_schema: List[Dict[str, Any]],
    ) -> None:
        if list(feature_names) != list(self.selected_feature_columns):
            raise ValueError(
                "Loaded dataset is incompatible with the imported model: selected feature names do not match"
            )

        incoming_schema_map = {
            str(item["name"]): item for item in feature_schema if isinstance(item, dict) and item.get("name")
        }
        for imported_feature in self.selected_feature_columns:
            imported_schema = self.feature_schema_map.get(imported_feature)
            incoming_schema = incoming_schema_map.get(imported_feature)
            if imported_schema is None or incoming_schema is None:
                raise ValueError(
                    f"Loaded dataset is incompatible with the imported model: missing schema for '{imported_feature}'"
                )
            if str(imported_schema.get("feature_type")) != str(incoming_schema.get("feature_type")):
                raise ValueError(
                    f"Loaded dataset is incompatible with the imported model: feature type mismatch for '{imported_feature}'"
                )
            if str(imported_schema.get("feature_type")) == "categorical":
                allowed = {
                    str(value)
                    for value in (imported_schema.get("categorical_options") or [])
                }
                incoming = {
                    str(value)
                    for value in (incoming_schema.get("categorical_options") or [])
                }
                if not incoming.issubset(allowed):
                    raise ValueError(
                        f"Loaded dataset is incompatible with the imported model: categorical values for '{imported_feature}' are not representable by the imported schema"
                    )

    def get_model_status(self) -> Dict[str, Any]:
        self.analytics_available = self._has_analytics_data()
        return {
            "is_trained": self.is_trained,
            "data_loaded": self.X_train is not None,
            "features": self.feature_names if self.feature_names else [],
            "feature_schema": self.feature_schema if self.feature_schema else [],
            "target_column": self.target_column,
            "selected_feature_columns": self.selected_feature_columns if self.selected_feature_columns else [],
            "dataset_id": self.active_dataset_id,
            "dataset_name": self.active_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None),
            "train_size": len(self.X_train) if self.X_train is not None else 0,
            "test_size": len(self.X_test) if self.X_test is not None else 0,
            "model_source": self.model_source,
            "analytics_available": self.analytics_available,
            "shape_functions_available": self._shape_functions_available(),
            "imported_artifact_version": self.imported_artifact_version,
            "comparison_available": self.comparison_is_trained,
            "comparison_data_loaded": self.comparison_X_train is not None,
            "comparison_is_trained": self.comparison_is_trained,
            "comparison_dataset_name": self.comparison_dataset_name
            or (
                Path(self.comparison_dataset_path).name
                if self.comparison_dataset_path
                else None
            ),
            "comparison_train_size": (
                len(self.comparison_X_train)
                if self.comparison_X_train is not None
                else 0
            ),
            "primary_n_estimators": self.primary_n_estimators,
        }

    def _reset_runtime_state(self) -> None:
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.is_trained = False
        self.feature_names = []
        self.selected_feature_columns = []
        self.cat_features = []
        self.num_features = []
        self.feature_schema = []
        self.feature_schema_map = {}
        self.model_source = "trained"
        self.imported_artifact_version = None
        self.analytics_available = False
        self.primary_n_estimators = None
        self.original_shape_functions = {}
        self.shape_function_offsets = {}
        self.target_column = None
        self.active_dataset_id = None
        self.active_dataset_path = None
        self.active_dataset_name = None
        self.feature_chart_settings = {}
        self._clear_comparison_state()

    def reset_all_data(self) -> Dict[str, int]:
        """
        Reset ML runtime state and remove persisted uploaded dataset artifacts.

        This keeps the built-in fallback dataset (bike.csv) untouched, but clears
        uploaded CSV files, active dataset metadata, and extracted feature state.
        """
        deleted_dataset_files = 0
        deleted_dataset_dirs = 0
        deleted_metadata_files = 0

        if self.datasets_dir.exists():
            for entry in self.datasets_dir.iterdir():
                if entry.is_file() or entry.is_symlink():
                    entry.unlink(missing_ok=True)
                    deleted_dataset_files += 1
                elif entry.is_dir():
                    shutil.rmtree(entry)
                    deleted_dataset_dirs += 1

        if self.active_dataset_file.exists():
            self.active_dataset_file.unlink(missing_ok=True)
            deleted_metadata_files += 1

        self._reset_runtime_state()
        self.data_store_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        return {
            "deleted_dataset_files": deleted_dataset_files,
            "deleted_dataset_dirs": deleted_dataset_dirs,
            "deleted_metadata_files": deleted_metadata_files,
        }

    def load_data(
        self,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load and prepare the active dataset."""
        previous_dataset_path = self.active_dataset_path
        previous_target = self.target_column
        previous_selected_features = list(self.selected_feature_columns)
        existing_model = self.model
        existing_shape_functions = dict(self.original_shape_functions)
        existing_model_source = self.model_source
        existing_artifact_version = self.imported_artifact_version
        existing_chart_settings = dict(self.feature_chart_settings)
        previous_runtime_state = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "df": self.df,
            "is_trained": self.is_trained,
            "feature_names": list(self.feature_names),
            "selected_feature_columns": list(self.selected_feature_columns),
            "cat_features": list(self.cat_features),
            "num_features": list(self.num_features),
            "target_column": self.target_column,
            "feature_schema": list(self.feature_schema),
            "feature_schema_map": dict(self.feature_schema_map),
            "model_source": self.model_source,
            "imported_artifact_version": self.imported_artifact_version,
            "analytics_available": self.analytics_available,
            "original_shape_functions": dict(self.original_shape_functions),
            "shape_function_offsets": {
                feature: dict(offsets)
                for feature, offsets in self.shape_function_offsets.items()
            },
            "active_dataset_id": self.active_dataset_id,
            "active_dataset_path": self.active_dataset_path,
            "active_dataset_name": self.active_dataset_name,
            "feature_chart_settings": dict(self.feature_chart_settings),
        }

        resolved_path, resolved_dataset_id = self._resolve_dataset_path(dataset_id)
        resolved_path_str = str(resolved_path)

        # Switch to a different dataset file: discard previous chart settings.
        if previous_dataset_path and previous_dataset_path != resolved_path_str:
            self.feature_chart_settings = {}
        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            df_loaded,
            resolved_target,
            selected_features,
            cat_features,
            num_features,
        ) = prepare_training_data(
            csv_path=str(resolved_path),
            target_column=target_column,
            feature_columns=feature_columns,
        )

        X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        rebuilt_feature_schema = self._build_feature_schema_for(
            list(X_train.columns),
            list(num_features),
            X_all,
        )
        rebuilt_feature_schema_map = {item["name"]: item for item in rebuilt_feature_schema}

        if existing_model_source == "imported" and existing_model is not None and self.is_trained:
            try:
                self._validate_dataset_compatibility(
                    feature_names=list(selected_features),
                    feature_schema=rebuilt_feature_schema,
                )
            except Exception:
                for key, value in previous_runtime_state.items():
                    setattr(self, key, value)
                raise

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = preprocessor
        self.df = df_loaded
        self.feature_names = list(X_train.columns)
        self.selected_feature_columns = list(selected_features)
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.target_column = resolved_target

        if existing_model_source == "imported" and existing_model is not None and previous_runtime_state["is_trained"]:
            self.model = existing_model
            self.is_trained = True
            self.model_source = existing_model_source
            self.imported_artifact_version = existing_artifact_version
            self.analytics_available = True
            self.feature_schema = list(self.feature_schema)
            self.feature_schema_map = dict(self.feature_schema_map)
            self.original_shape_functions = existing_shape_functions
            self.shape_function_offsets = {}
            self.feature_chart_settings = {
                feature: setting
                for feature, setting in existing_chart_settings.items()
                if feature in self.feature_names
            }
        else:
            # Any dataset load invalidates a trained local model and shape offsets.
            self.model = None
            self.is_trained = False
            self.model_source = "trained"
            self.imported_artifact_version = None
            self.analytics_available = False
            self.original_shape_functions = {}
            self.shape_function_offsets = {}
            self.feature_schema = rebuilt_feature_schema
            self.feature_schema_map = rebuilt_feature_schema_map
            self.feature_chart_settings = {
                feature: setting
                for feature, setting in self.feature_chart_settings.items()
                if feature in self.feature_names
            }

        self.active_dataset_id = resolved_dataset_id
        self.active_dataset_path = resolved_path_str
        if dataset_name:
            self.active_dataset_name = Path(str(dataset_name)).name
        else:
            self.active_dataset_name = Path(self.active_dataset_path).name if self.active_dataset_path else None
        self._persist_active_dataset_metadata()

        self._clear_comparison_state()

        dataset_changed = (
            previous_dataset_path != self.active_dataset_path
            or previous_target != self.target_column
            or previous_selected_features != self.selected_feature_columns
        )

        return {
            "total_records": len(self.df),
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "features": self.feature_names,
            "feature_schema": self.feature_schema,
            "target_column": self.target_column,
            "selected_feature_columns": self.selected_feature_columns,
            "dataset_id": self.active_dataset_id,
            "dataset_name": self.active_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None),
            "dataset_changed": dataset_changed,
        }

    def _validate_comparison_dataset_selection(
        self,
        *,
        target_column: str,
        feature_columns: List[str],
    ) -> None:
        if not self.is_trained or not self.primary_n_estimators:
            raise ValueError(
                "Train the primary model first so the comparison model can inherit its estimator count"
            )
        if target_column != self.target_column:
            raise ValueError(
                "Comparison dataset target column must match the primary dataset target column exactly"
            )
        if list(feature_columns) != list(self.selected_feature_columns):
            raise ValueError(
                "Comparison dataset feature columns must match the primary dataset selected feature columns exactly"
            )

    def _validate_comparison_feature_schema(
        self,
        feature_schema: List[Dict[str, Any]],
    ) -> None:
        """Validate comparison dataset feature compatibility without requiring identical derived defaults."""
        incoming_schema_map = {
            str(item["name"]): item
            for item in feature_schema
            if isinstance(item, dict) and item.get("name")
        }
        primary_schema_map = {
            str(item["name"]): item
            for item in self.feature_schema
            if isinstance(item, dict) and item.get("name")
        }

        for feature_name in self.selected_feature_columns:
            primary_schema = primary_schema_map.get(feature_name)
            comparison_schema = incoming_schema_map.get(feature_name)
            if primary_schema is None or comparison_schema is None:
                raise ValueError(
                    f"Comparison dataset schema is incompatible: missing schema for '{feature_name}'"
                )

            primary_type = str(primary_schema.get("feature_type"))
            comparison_type = str(comparison_schema.get("feature_type"))
            if primary_type != comparison_type:
                raise ValueError(
                    f"Comparison dataset schema is incompatible: feature type mismatch for '{feature_name}'"
                )

            if primary_type == "categorical":
                primary_options = {
                    str(value)
                    for value in (primary_schema.get("categorical_options") or [])
                }
                comparison_options = {
                    str(value)
                    for value in (comparison_schema.get("categorical_options") or [])
                }
                if not comparison_options.issubset(primary_options):
                    raise ValueError(
                        f"Comparison dataset schema is incompatible: categorical values for '{feature_name}' are not representable by the primary dataset schema"
                    )

    def load_comparison_data(
        self,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load and validate a second dataset for shape-function comparison."""
        resolved_path, resolved_dataset_id = self._resolve_dataset_path(dataset_id)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            df_loaded,
            resolved_target,
            selected_features,
            cat_features,
            num_features,
        ) = prepare_training_data(
            csv_path=str(resolved_path),
            target_column=target_column,
            feature_columns=feature_columns,
        )

        selected_features = list(selected_features)
        self._validate_comparison_dataset_selection(
            target_column=resolved_target,
            feature_columns=selected_features,
        )

        X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        rebuilt_feature_schema = self._build_feature_schema_for(
            list(X_train.columns),
            list(num_features),
            X_all,
        )
        self._validate_comparison_feature_schema(rebuilt_feature_schema)

        self.comparison_model = None
        self.comparison_is_trained = False
        self.comparison_original_shape_functions = {}
        self.comparison_shape_function_offsets = {}
        self.comparison_X_train = X_train
        self.comparison_X_test = X_test
        self.comparison_y_train = y_train
        self.comparison_y_test = y_test
        self.comparison_preprocessor = preprocessor
        self.comparison_df = df_loaded
        self.comparison_feature_names = list(X_train.columns)
        self.comparison_selected_feature_columns = selected_features
        self.comparison_cat_features = list(cat_features)
        self.comparison_num_features = list(num_features)
        self.comparison_target_column = resolved_target
        self.comparison_feature_schema = rebuilt_feature_schema
        self.comparison_feature_schema_map = {
            item["name"]: item for item in rebuilt_feature_schema
        }
        self.comparison_dataset_id = resolved_dataset_id
        self.comparison_dataset_path = str(resolved_path)
        self.comparison_dataset_name = (
            Path(str(dataset_name)).name
            if dataset_name
            else Path(str(resolved_path)).name
        )

        return {
            "total_records": len(df_loaded),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": list(X_train.columns),
            "target_column": resolved_target,
            "selected_feature_columns": selected_features,
            "dataset_id": self.comparison_dataset_id,
            "dataset_name": self.comparison_dataset_name,
            "primary_n_estimators": self.primary_n_estimators,
        }

    def train_comparison_model(self) -> int:
        """Train the comparison model using the primary model's estimator count."""
        if not self.is_trained or not self.primary_n_estimators:
            raise ValueError(
                "Train the primary model first so the comparison model can inherit its estimator count"
            )
        if self.comparison_X_train is None or self.comparison_y_train is None:
            raise ValueError("Load a comparison dataset first")

        self.comparison_model = IGANN(
            task="regression",
            n_estimators=self.primary_n_estimators,
            verbose=0,
            scale_y=True,
        )
        self.comparison_model.fit(self.comparison_X_train, self.comparison_y_train)
        self.comparison_is_trained = True
        self.comparison_original_shape_functions = {}
        self.comparison_shape_function_offsets = {}
        return int(self.primary_n_estimators)

    # ==================== Model lifecycle ====================

    def train_model(self, n_estimators: int = 100) -> Dict[str, float]:
        """Train the IGANN model."""
        if self.X_train is None:
            self.load_data()

        self.model = IGANN(
            task="regression",
            n_estimators=n_estimators,
            verbose=0,
            scale_y=True,
        )
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        self.model_source = "trained"
        self.imported_artifact_version = None
        self.analytics_available = self._has_analytics_data()
        self.primary_n_estimators = int(n_estimators)
        self.original_shape_functions = {}
        self.shape_function_offsets = {}
        self._clear_comparison_state()

        metrics = self.evaluate_model()
        return metrics

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        self._require_analytics_data()

        y_pred = self.model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "model_type": "IGANN",
        }

    # ==================== Prediction helpers ====================

    def _validate_and_normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction payload against loaded feature schema."""
        if not self.feature_schema_map:
            raise ValueError("Feature schema is not available yet")

        provided_keys = set(features.keys())
        expected_keys = set(self.feature_names)
        missing = sorted(expected_keys - provided_keys)
        extra = sorted(provided_keys - expected_keys)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(f"Prediction features do not match dataset schema ({', '.join(details)})")

        normalized: Dict[str, Any] = {}
        for feature_name in self.feature_names:
            value = features[feature_name]
            schema = self.feature_schema_map.get(feature_name, {})
            feature_type = schema.get("feature_type")

            if feature_type == "numeric":
                try:
                    normalized[feature_name] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature '{feature_name}' must be numeric") from exc
            else:
                value_str = str(value)
                valid_options = schema.get("categorical_options") or []
                if valid_options and value_str not in valid_options:
                    raise ValueError(
                        f"Feature '{feature_name}' value '{value_str}' is not in allowed options"
                    )
                normalized[feature_name] = value_str

        return normalized

    def _build_input_df(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Build a single-row DataFrame from dynamic prediction features."""
        normalized = self._validate_and_normalize_features(features)
        input_df = pd.DataFrame(
            [{feature_name: normalized[feature_name] for feature_name in self.feature_names}]
        )
        for cat_feature in self.cat_features:
            if cat_feature in input_df.columns:
                input_df[cat_feature] = input_df[cat_feature].astype(str)
        for num_feature in self.num_features:
            if num_feature in input_df.columns:
                input_df[num_feature] = pd.to_numeric(input_df[num_feature], errors="coerce")
        return input_df

    # ==================== Prediction APIs ====================

    def predict(self, features: Dict[str, Any]) -> float:
        """Make a single prediction, applying current shape function offsets."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        input_df = self._build_input_df(features)
        base_prediction = float(self.model.predict(input_df)[0])

        total_offset = 0.0
        for feature_name in self.shape_function_offsets:
            if feature_name in input_df.columns:
                value = input_df[feature_name].iloc[0]
                total_offset += self._get_offset_for_value(feature_name, value)

        return base_prediction + total_offset

    def predict_with_offsets(
        self,
        features: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
    ) -> float:
        """Make a single prediction using provided shape-function offsets."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        input_df = self._build_input_df(features)
        base_prediction = float(self.model.predict(input_df)[0])

        original_offsets = self.shape_function_offsets
        self.shape_function_offsets = offsets

        total_offset = 0.0
        for feature_name in offsets:
            if feature_name in input_df.columns:
                value = input_df[feature_name].iloc[0]
                total_offset += self._get_offset_for_value(feature_name, value)

        self.shape_function_offsets = original_offsets
        return base_prediction + total_offset

    def batch_predict(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Make batch predictions."""
        return [self.predict(features) for features in features_list]

    # ==================== Shape functions ====================

    def get_shape_functions(self) -> List[Dict[str, Any]]:
        """Get shape function data for all features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        if self.model_source == "imported" and self.original_shape_functions:
            return list(self.original_shape_functions.values())

        shape_functions = []
        for idx, feature_name in enumerate(self.feature_names):
            try:
                shape_data = self._extract_shape_function(feature_name, idx)
                shape_functions.append(shape_data)
                self.original_shape_functions[feature_name] = shape_data
            except Exception as exc:
                print(f"Error extracting shape function for {feature_name}: {exc}")
                continue

        self.shape_function_offsets = {}
        return shape_functions

    def get_comparison_shape_functions(self) -> List[Dict[str, Any]]:
        """Get shape function data for the comparison model, aligned to primary features."""
        if not self.comparison_is_trained or self.comparison_model is None:
            return []

        snapshot = self._snapshot_runtime_state()
        try:
            self.model = self.comparison_model
            self.preprocessor = self.comparison_preprocessor
            self.X_train = self.comparison_X_train
            self.X_test = self.comparison_X_test
            self.y_train = self.comparison_y_train
            self.y_test = self.comparison_y_test
            self.df = self.comparison_df
            self.is_trained = self.comparison_is_trained
            self.feature_names = list(self.comparison_feature_names)
            self.selected_feature_columns = list(
                self.comparison_selected_feature_columns
            )
            self.cat_features = list(self.comparison_cat_features)
            self.num_features = list(self.comparison_num_features)
            self.target_column = self.comparison_target_column
            self.feature_schema = list(self.comparison_feature_schema)
            self.feature_schema_map = dict(self.comparison_feature_schema_map)
            self.original_shape_functions = dict(
                self.comparison_original_shape_functions
            )
            self.shape_function_offsets = {
                feature: dict(offsets)
                for feature, offsets in self.comparison_shape_function_offsets.items()
            }
            self.model_source = "trained"
            self.imported_artifact_version = None
            self.analytics_available = False

            shape_functions = self.get_shape_functions()
            self.comparison_original_shape_functions = {
                shape_function["feature_name"]: shape_function
                for shape_function in shape_functions
            }
            self.comparison_shape_function_offsets = {}
            return shape_functions
        finally:
            self._restore_runtime_state(snapshot)

    def _build_shape_baseline(self) -> Dict[str, Any]:
        baseline: Dict[str, Any] = {}
        for num_feat in self.num_features:
            baseline[num_feat] = float(pd.to_numeric(self.X_train[num_feat], errors="coerce").mean())
        for cat_feat in self.cat_features:
            mode = self.X_train[cat_feat].astype(str).mode()
            baseline[cat_feat] = str(mode.iloc[0]) if not mode.empty else "Unknown"
        return baseline

    def _extract_shape_function(self, feature_name: str, feature_idx: int) -> Dict[str, Any]:
        """Extract shape function data for a single feature."""
        chart_setting = self.get_feature_chart_setting(feature_name)
        is_categorical = chart_setting.get("chart_feature_type") == "categorical"
        value_labels = chart_setting.get("value_labels", {}) or {}
        baseline = self._build_shape_baseline()

        if is_categorical:
            raw_x_values = self._get_chart_x_values(
                feature_name,
                treat_as_categorical=True,
            )
            shape_values = []
            effective_x_values: List[str] = []
            for raw_value in raw_x_values:
                sample_data = baseline.copy()
                if feature_name in self.num_features:
                    try:
                        sample_data[feature_name] = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                else:
                    sample_data[feature_name] = raw_value
                sample = pd.DataFrame([sample_data])[self.feature_names]

                for cat_feat in self.cat_features:
                    sample[cat_feat] = sample[cat_feat].astype(str)
                for num_feat in self.num_features:
                    sample[num_feat] = sample[num_feat].astype(float)

                pred = self.model.predict(sample)
                pred_val = float(pred[0]) if hasattr(pred, "__iter__") else float(pred)
                shape_values.append(pred_val)
                effective_x_values.append(raw_value)

            if not effective_x_values:
                is_categorical = False

            if is_categorical:
                mean_val = np.mean(shape_values)
                shape_values = [value - mean_val for value in shape_values]
                x_tick_labels = [
                    str(value_labels.get(raw_value, raw_value))
                    for raw_value in effective_x_values
                ]

                return {
                    "feature_name": feature_name,
                    "x_values": effective_x_values,
                    "x_tick_labels": x_tick_labels,
                    "y_values": shape_values,
                    "feature_type": "categorical",
                    "chart_config": chart_setting,
                }

        if feature_name in self.cat_features:
            numeric_pairs = self._get_numeric_chart_values_for_categorical_feature(
                feature_name,
            )
            if numeric_pairs:
                x_values_numeric: List[float] = []
                shape_values = []
                for raw_value, numeric_x in numeric_pairs:
                    sample_data = baseline.copy()
                    sample_data[feature_name] = raw_value
                    sample = pd.DataFrame([sample_data])[self.feature_names]

                    for cat_feat in self.cat_features:
                        sample[cat_feat] = sample[cat_feat].astype(str)
                    for num_feat in self.num_features:
                        sample[num_feat] = sample[num_feat].astype(float)

                    pred = self.model.predict(sample)
                    pred_val = float(pred[0]) if hasattr(pred, "__iter__") else float(pred)
                    shape_values.append(pred_val)
                    x_values_numeric.append(float(numeric_x))

                mean_val = np.mean(shape_values)
                shape_values = [value - mean_val for value in shape_values]

                return {
                    "feature_name": feature_name,
                    "x_values": x_values_numeric,
                    "y_values": shape_values,
                    "feature_type": "numeric",
                    "chart_config": chart_setting,
                }

        min_val = float(pd.to_numeric(self.X_train[feature_name], errors="coerce").min())
        max_val = float(pd.to_numeric(self.X_train[feature_name], errors="coerce").max())
        if min_val == max_val:
            x_range = np.array([min_val])
        else:
            x_range = np.linspace(min_val, max_val, 30)

        shape_values = []
        for x_val in x_range:
            sample_data = baseline.copy()
            sample_data[feature_name] = float(x_val)
            sample = pd.DataFrame([sample_data])[self.feature_names]

            for cat_feat in self.cat_features:
                sample[cat_feat] = sample[cat_feat].astype(str)
            for num_feat in self.num_features:
                sample[num_feat] = sample[num_feat].astype(float)

            pred = self.model.predict(sample)
            pred_val = float(pred[0]) if hasattr(pred, "__iter__") else float(pred)
            shape_values.append(pred_val)

        mean_val = np.mean(shape_values)
        shape_values = [value - mean_val for value in shape_values]

        return {
            "feature_name": feature_name,
            "x_values": x_range.tolist(),
            "y_values": shape_values,
            "feature_type": "numeric",
            "chart_config": chart_setting,
        }

    # ==================== Dataset analytics ====================

    def get_predictions_vs_actual(self) -> Dict[str, List[float]]:
        """Get predictions vs actual values for visualization."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        self._require_analytics_data()

        y_pred = self.model.predict(self.X_test)
        y_actual = self.y_test.values.flatten().tolist()
        return {
            "predicted": [float(value) for value in y_pred],
            "actual": y_actual,
        }

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the loaded dataset."""
        if self.df is None:
            raise ValueError("Data not loaded yet")

        target_values = pd.to_numeric(self.df[self.target_column], errors="coerce")
        return {
            "total_records": len(self.df),
            "features": self.feature_names,
            "numeric_features": self.num_features,
            "categorical_features": self.cat_features,
            "target_column": self.target_column,
            "target_stats": {
                "mean": float(target_values.mean()),
                "std": float(target_values.std()),
                "min": float(target_values.min()),
                "max": float(target_values.max()),
            },
        }

    def get_feature_distributions(self) -> Dict[str, Any]:
        """Get feature distributions for visualization."""
        if self.X_train is None:
            raise ValueError("Data not loaded yet")

        distributions: Dict[str, Any] = {}
        for feature in self.num_features:
            values = pd.to_numeric(self.X_train[feature], errors="coerce").dropna().tolist()
            distributions[feature] = {
                "type": "numeric",
                "values": values[:1000],
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
            }

        for feature in self.cat_features:
            value_counts = self.X_train[feature].astype(str).value_counts().to_dict()
            distributions[feature] = {
                "type": "categorical",
                "counts": {str(key): int(value) for key, value in value_counts.items()},
            }

        return distributions

    def get_hourly_pattern(self) -> Dict[str, Any]:
        """Get hourly target pattern when standard bike columns exist."""
        if self.df is None:
            raise ValueError("Data not loaded yet")

        if "hr" not in self.df.columns or self.target_column not in self.df.columns:
            return {
                "available": False,
                "hours": [],
                "avg_rentals": [],
                "message": "Hourly pattern is not available for this dataset.",
            }

        hourly_avg = self.df.groupby("hr")[self.target_column].mean()
        return {
            "available": True,
            "hours": hourly_avg.index.tolist(),
            "avg_rentals": hourly_avg.values.tolist(),
        }

    # ==================== Interactive edits ====================

    @staticmethod
    def _interpolate_curve_value(
        x_values: List[Any],
        y_values: List[Any],
        target_x: float,
    ) -> float:
        """Linearly interpolate y on a curve at target_x."""
        if not x_values or not y_values or len(x_values) != len(y_values):
            return 0.0

        try:
            pairs = sorted(
                [(float(x), float(y)) for x, y in zip(x_values, y_values)],
                key=lambda item: item[0],
            )
        except (TypeError, ValueError):
            return 0.0

        if not pairs:
            return 0.0
        if len(pairs) == 1:
            return pairs[0][1]

        if target_x <= pairs[0][0]:
            return pairs[0][1]
        if target_x >= pairs[-1][0]:
            return pairs[-1][1]

        for idx in range(len(pairs) - 1):
            left_x, left_y = pairs[idx]
            right_x, right_y = pairs[idx + 1]
            if left_x <= target_x <= right_x:
                span = right_x - left_x
                if span == 0:
                    return left_y
                ratio = (target_x - left_x) / span
                return left_y + ratio * (right_y - left_y)

        return pairs[-1][1]

    @staticmethod
    def _encode_numeric_storage_key(x_value: float) -> str:
        """Encode numeric x values with an explicit prefix for persistence."""
        return f"x:{x_value:.12g}"

    @staticmethod
    def _decode_numeric_storage_key(raw_key: Any, original_x: List[Any]) -> Optional[float]:
        """
        Decode numeric storage key.

        Supported formats:
        - Legacy index (int / numeric string)
        - Explicit coordinate ("x:<float>")
        - Fallback direct float value
        """
        if isinstance(raw_key, (int, np.integer)):
            idx = int(raw_key)
            if 0 <= idx < len(original_x):
                return float(original_x[idx])
            return None

        if isinstance(raw_key, (float, np.floating)):
            return float(raw_key)

        raw_str = str(raw_key).strip()
        if not raw_str:
            return None

        if raw_str.startswith("x:"):
            try:
                return float(raw_str[2:])
            except ValueError:
                return None

        try:
            idx = int(raw_str)
            if 0 <= idx < len(original_x):
                return float(original_x[idx])
        except ValueError:
            pass

        try:
            return float(raw_str)
        except ValueError:
            return None

    def _decode_numeric_offsets_for_feature(
        self,
        feature_name: str,
        offsets: Dict[Any, float],
    ) -> List[Tuple[float, float]]:
        """Convert stored/legacy numeric offsets into sorted (x, offset) points."""
        original = self.original_shape_functions.get(feature_name, {})
        original_x = original.get("x_values", [])

        point_map: Dict[float, float] = {}
        for raw_key, raw_offset in (offsets or {}).items():
            x_val = self._decode_numeric_storage_key(raw_key, original_x)
            if x_val is None:
                continue
            try:
                point_map[float(x_val)] = float(raw_offset)
            except (TypeError, ValueError):
                continue

        return sorted(point_map.items(), key=lambda item: item[0])

    def load_shape_function_offsets_from_storage(
        self,
        storage_edits: List[Dict[str, Any]],
    ) -> None:
        """Load persisted edits into runtime offsets (supports legacy and new formats)."""
        self.shape_function_offsets = {}

        for stored_sf in storage_edits:
            feature_name = stored_sf.get("feature_name")
            feature_type = stored_sf.get("feature_type")
            stored_points = stored_sf.get("edited_points", [])

            if not feature_name or feature_name not in self.original_shape_functions:
                continue

            self.shape_function_offsets[feature_name] = {}
            original_x = self.original_shape_functions[feature_name]["x_values"]

            for stored_point in stored_points:
                x_val = stored_point.get("x_value")
                offset = stored_point.get("y_value", 0.0)
                try:
                    offset_float = float(offset)
                except (TypeError, ValueError):
                    continue

                if feature_type == "categorical":
                    self.shape_function_offsets[feature_name][str(x_val)] = offset_float
                else:
                    decoded_x = self._decode_numeric_storage_key(x_val, original_x)
                    if decoded_x is None:
                        continue
                    self.shape_function_offsets[feature_name][float(decoded_x)] = offset_float

    def update_shape_functions(self, edited_shape_functions: List[Dict[str, Any]]) -> None:
        """Update shape function offsets based on user edits."""
        self.shape_function_offsets = {}

        for edited_sf in edited_shape_functions:
            feature_name = edited_sf["feature_name"]
            feature_type = edited_sf["feature_type"]
            edited_points = edited_sf["edited_points"]

            if feature_name not in self.original_shape_functions:
                continue

            original = self.original_shape_functions[feature_name]
            original_x = original["x_values"]
            original_y = original["y_values"]
            self.shape_function_offsets[feature_name] = {}

            for edited_point in edited_points:
                x_val = edited_point["x_value"]
                new_y = edited_point["y_value"]

                if feature_type == "categorical":
                    x_str = str(x_val)
                    if x_str in original_x:
                        idx = original_x.index(x_str)
                        original_y_val = original_y[idx]
                        offset = new_y - original_y_val
                        self.shape_function_offsets[feature_name][x_str] = offset
                else:
                    x_float = float(x_val)
                    original_y_val = self._interpolate_curve_value(original_x, original_y, x_float)
                    offset = new_y - original_y_val
                    self.shape_function_offsets[feature_name][x_float] = offset

    def convert_edits_for_storage(
        self,
        edited_shape_functions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert edited shape functions to storage format.

        Numeric features: x_value -> encoded coordinate key, y_value -> offset.
        Categorical features: x_value string, y_value -> offset.
        """
        result = []
        for edited_sf in edited_shape_functions:
            feature_name = edited_sf["feature_name"]
            feature_type = edited_sf["feature_type"]
            edited_points = edited_sf["edited_points"]

            if feature_name not in self.original_shape_functions:
                continue

            original = self.original_shape_functions[feature_name]
            original_x = original["x_values"]
            original_y = original["y_values"]

            converted_points = []
            for edited_point in edited_points:
                x_val = edited_point["x_value"]
                new_y = edited_point["y_value"]
                weight = edited_point.get("weight", 0.5)
                message = edited_point.get("message", "")

                if feature_type == "categorical":
                    x_str = str(x_val)
                    if x_str in original_x:
                        idx = original_x.index(x_str)
                        offset = new_y - original_y[idx]
                        converted_points.append(
                            {
                                "x_value": x_str,
                                "y_value": offset,
                                "weight": weight,
                                "message": message,
                            }
                        )
                else:
                    x_float = float(x_val)
                    offset = new_y - self._interpolate_curve_value(original_x, original_y, x_float)
                    converted_points.append(
                        {
                            "x_value": self._encode_numeric_storage_key(x_float),
                            "y_value": offset,
                            "weight": weight,
                            "message": message,
                        }
                    )

            if converted_points:
                result.append(
                    {
                        "feature_name": feature_name,
                        "feature_type": feature_type,
                        "edited_points": converted_points,
                    }
                )
        return result

    def convert_storage_to_display_format(
        self,
        storage_edits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert stored edits back to display format for frontend rendering."""
        result = []
        for stored_sf in storage_edits:
            feature_name = stored_sf["feature_name"]
            feature_type = stored_sf["feature_type"]
            stored_points = stored_sf.get("edited_points", [])

            if feature_name not in self.original_shape_functions:
                continue

            original = self.original_shape_functions[feature_name]
            original_x = original["x_values"]
            original_y = original["y_values"]

            display_points = []
            for stored_point in stored_points:
                x_val = stored_point["x_value"]
                offset = stored_point["y_value"]

                if feature_type == "categorical":
                    x_str = str(x_val)
                    if x_str in original_x:
                        idx = original_x.index(x_str)
                        display_points.append(
                            {
                                "x_value": x_str,
                                "y_value": original_y[idx] + offset,
                            }
                        )
                else:
                    decoded_x = self._decode_numeric_storage_key(x_val, original_x)
                    if decoded_x is None:
                        continue
                    base_y = self._interpolate_curve_value(original_x, original_y, decoded_x)
                    display_points.append(
                        {
                            "x_value": float(decoded_x),
                            "y_value": base_y + offset,
                        }
                    )

            if display_points:
                result.append(
                    {
                        "feature_name": feature_name,
                        "feature_type": feature_type,
                        "edited_points": display_points,
                    }
                )
        return result

    def _get_offset_for_value(self, feature_name: str, value: Any) -> float:
        """Get interpolated offset for a feature value."""
        if feature_name not in self.shape_function_offsets:
            return 0.0

        offsets = self.shape_function_offsets[feature_name]
        return self.get_offset_for_feature_value(feature_name, value, offsets)

    def get_offset_for_feature_value(
        self,
        feature_name: str,
        value: Any,
        offsets: Optional[Dict[Any, float]] = None,
    ) -> float:
        """Get interpolated offset for a feature value from a given offsets map."""
        offsets = offsets if offsets is not None else self.shape_function_offsets.get(feature_name, {})
        if not offsets:
            return 0.0

        is_categorical_chart = self._is_feature_categorical_for_chart(feature_name)
        if is_categorical_chart:
            candidate_keys = [
                self._stringify_chart_value(value),
                str(value),
            ]
            for raw_key in candidate_keys:
                if raw_key in offsets:
                    try:
                        return float(offsets.get(raw_key, 0.0))
                    except (TypeError, ValueError):
                        return 0.0

            # Backward compatibility: if old numeric-style keys exist, use exact numeric match.
            points = self._decode_numeric_offsets_for_feature(feature_name, offsets)
            if points:
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    return 0.0
                for x_val, x_offset in points:
                    if abs(x_val - value_float) < 1e-9:
                        return float(x_offset)
            return 0.0

        # Numeric chart mode on originally categorical features:
        # allow exact string key lookup (from previously stored categorical edits)
        # before attempting numeric interpolation.
        if feature_name in self.cat_features:
            candidate_keys = [
                self._stringify_chart_value(value),
                str(value),
            ]
            for raw_key in candidate_keys:
                if raw_key in offsets:
                    try:
                        return float(offsets.get(raw_key, 0.0))
                    except (TypeError, ValueError):
                        return 0.0

        points = self._decode_numeric_offsets_for_feature(feature_name, offsets)
        if not points:
            return 0.0

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return 0.0

        if len(points) == 1:
            return float(points[0][1])

        x_positions = [point[0] for point in points]
        offset_values = [point[1] for point in points]

        if value_float <= x_positions[0]:
            return float(offset_values[0])
        if value_float >= x_positions[-1]:
            return float(offset_values[-1])

        for idx in range(len(points) - 1):
            left_x, left_offset = points[idx]
            right_x, right_offset = points[idx + 1]
            if left_x <= value_float <= right_x:
                span = right_x - left_x
                if span == 0:
                    return float(left_offset)
                t_val = (value_float - left_x) / span
                return float(left_offset + t_val * (right_offset - left_offset))

        return 0.0

    def predict_interactive(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with interactive shape function modifications."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        base_predictions = self.model.predict(X)
        offsets = np.zeros(len(X))

        for feature_name in self.shape_function_offsets:
            if feature_name in X.columns:
                for idx, value in enumerate(X[feature_name].values):
                    offsets[idx] += self._get_offset_for_value(feature_name, value)

        return base_predictions + offsets

    def get_predictions_comparison(self) -> Dict[str, Any]:
        """Get comparison between original and interactive predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        self._require_analytics_data()

        y_pred_original = self.model.predict(self.X_test)
        y_pred_interactive = self.predict_interactive(self.X_test)
        y_actual = self.y_test.values.flatten()

        original_rmse = root_mean_squared_error(y_actual, y_pred_original)
        original_mae = mean_absolute_error(y_actual, y_pred_original)
        interactive_rmse = root_mean_squared_error(y_actual, y_pred_interactive)
        interactive_mae = mean_absolute_error(y_actual, y_pred_interactive)

        return {
            "original_predictions": [float(value) for value in y_pred_original],
            "interactive_predictions": [float(value) for value in y_pred_interactive],
            "actual_values": [float(value) for value in y_actual],
            "metrics": {
                "original_rmse": float(original_rmse),
                "original_mae": float(original_mae),
                "interactive_rmse": float(interactive_rmse),
                "interactive_mae": float(interactive_mae),
            },
        }


# Create singleton instance
ml_service = MLService()
