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
        self.model: Optional[IGANN] = None
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

        # Store original shape functions for interactive editing
        self.original_shape_functions: Dict[str, Dict[str, Any]] = {}
        self.shape_function_offsets: Dict[str, Dict[Any, float]] = {}

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
        unique_ratio = unique_count / max(len(series), 1)
        return unique_count <= 12 or (unique_count <= 24 and unique_ratio <= 0.05)

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

    def _build_feature_schema(self, X_all: pd.DataFrame) -> List[Dict[str, Any]]:
        schema: List[Dict[str, Any]] = []
        for feature_name in self.feature_names:
            if feature_name in self.num_features:
                numeric_series = pd.to_numeric(X_all[feature_name], errors="coerce")
                min_val = float(numeric_series.min())
                max_val = float(numeric_series.max())
                default_val = float(numeric_series.median())

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

                schema.append(
                    {
                        "name": feature_name,
                        "feature_type": "categorical",
                        "default_value": default_val,
                        "categorical_options": unique_values,
                    }
                )

        return schema

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
        self.original_shape_functions = {}
        self.shape_function_offsets = {}
        self.target_column = None
        self.active_dataset_id = None
        self.active_dataset_path = None
        self.active_dataset_name = None
        self.feature_chart_settings = {}

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

        # Any dataset load invalidates trained state and shape offsets.
        self.model = None
        self.is_trained = False
        self.original_shape_functions = {}
        self.shape_function_offsets = {}

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

        # Keep only settings for currently selected features.
        self.feature_chart_settings = {
            feature: setting
            for feature, setting in self.feature_chart_settings.items()
            if feature in self.feature_names
        }

        X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        self.feature_schema = self._build_feature_schema(X_all)
        self.feature_schema_map = {item["name"]: item for item in self.feature_schema}

        self.active_dataset_id = resolved_dataset_id
        self.active_dataset_path = resolved_path_str
        if dataset_name:
            self.active_dataset_name = Path(str(dataset_name)).name
        else:
            self.active_dataset_name = Path(self.active_dataset_path).name if self.active_dataset_path else None
        self._persist_active_dataset_metadata()

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

        metrics = self.evaluate_model()
        return metrics

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

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
        if self.X_train is None:
            raise ValueError("Data not loaded yet")

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
