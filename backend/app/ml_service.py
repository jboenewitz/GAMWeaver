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

from .data_processing import (
    MISSING_CATEGORY_VALUE,
    prepare_training_data,
)

DATA_FILE_NAME = "bike.csv"
MAX_PREVIEW_ROWS = 500
SHAPE_DISTRIBUTION_DEFAULT_BIN_COUNT = 20
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
        self.missing_indicator_map: Dict[str, str] = {}
        self.numeric_missing_placeholder_values: Dict[str, float] = {}
        self.show_missing_bars = False
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
        self.comparison_missing_indicator_map: Dict[str, str] = {}
        self.comparison_numeric_missing_placeholder_values: Dict[str, float] = {}
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
        self._feature_dictionary_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._column_mapping_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

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
        self.comparison_missing_indicator_map = {}
        self.comparison_numeric_missing_placeholder_values = {}
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
            "missing_indicator_map": dict(self.missing_indicator_map),
            "numeric_missing_placeholder_values": dict(
                self.numeric_missing_placeholder_values
            ),
            "show_missing_bars": bool(self.show_missing_bars),
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
        show_missing_bars = data.get("show_missing_bars")
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
                normalized = self._normalize_stored_feature_chart_setting(raw_setting)
                if normalized:
                    sanitized[str(feature_name)] = normalized
            self.feature_chart_settings = sanitized
        if show_missing_bars is not None:
            self.show_missing_bars = bool(show_missing_bars)

    def _persist_active_dataset_metadata(self) -> None:
        payload = {
            "dataset_id": self.active_dataset_id,
            "dataset_path": self.active_dataset_path,
            "dataset_name": self.active_dataset_name,
            "target_column": self.target_column,
            "selected_feature_columns": self.selected_feature_columns,
            "feature_chart_settings": self.feature_chart_settings,
            "show_missing_bars": bool(self.show_missing_bars),
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
    def _split_dictionary_values(raw_value: Any) -> List[str]:
        if raw_value is None:
            return []
        text = str(raw_value).strip()
        if not text or text.lower() == "nan":
            return []
        return [part.strip() for part in text.split("|") if part.strip()]

    @staticmethod
    def _classify_feature_construction(
        transformation: str,
        source_labels: List[str],
    ) -> str:
        normalized_transformation = transformation.strip().lower()
        normalized_labels = " ".join(source_labels).strip().lower()

        if (
            "mean across item battery" in normalized_transformation
            or "itemmittelwert" in normalized_transformation
            or "mittelwert aus" in normalized_transformation
            or "row mean across item battery" in normalized_transformation
        ):
            return "item_mean"

        if "scale" in normalized_labels and "imputiert" in normalized_labels:
            return "iqb_scale"
        if "skala" in normalized_labels and "imputiert" in normalized_labels:
            return "iqb_scale"

        return "raw_source"

    @staticmethod
    def _extract_missing_value_handling(transformation: str) -> str:
        text = str(transformation or "").strip()
        if not text:
            return ""

        parts = [part.strip() for part in text.split(";") if part.strip()]
        if len(parts) > 1:
            return "; ".join(parts[1:])

        lower_text = text.lower()
        if "missing codes replaced with na" in lower_text:
            return "IQB special missing codes replaced with NA"
        if "sonderfehlwerte" in lower_text:
            return text

        return ""

    def _load_feature_provenance_from_path(
        self,
        dictionary_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        cache_key = str(dictionary_path)
        cached = self._feature_dictionary_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            dictionary_df = pd.read_csv(dictionary_path, dtype=str, encoding="utf-8-sig")
        except Exception:
            self._feature_dictionary_cache[cache_key] = {}
            return {}

        required_columns = {
            "output_column",
            "role",
            "category",
            "source_variables",
            "source_labels",
            "transformation",
            "selection_rationale",
        }
        if not required_columns.issubset(set(dictionary_df.columns)):
            self._feature_dictionary_cache[cache_key] = {}
            return {}

        response_option_map = self._build_response_option_map_for_dictionary(
            dictionary_path,
        )
        provenance_map: Dict[str, Dict[str, Any]] = {}
        for _, row in dictionary_df.iterrows():
            output_column = str(row.get("output_column", "") or "").strip()
            if not output_column:
                continue

            source_variables = self._split_dictionary_values(row.get("source_variables"))
            source_labels = self._split_dictionary_values(row.get("source_labels"))
            transformation = str(row.get("transformation", "") or "").strip()
            selection_rationale = str(row.get("selection_rationale", "") or "").strip()
            source_details = []
            common_response_options: Optional[List[Dict[str, Any]]] = None

            for index, variable in enumerate(source_variables):
                response_metadata = response_option_map.get(variable, {})
                detail = {
                    "variable": variable,
                    "label": source_labels[index] if index < len(source_labels) else "",
                    "response_options": _json_safe(
                        response_metadata.get("response_options", [])
                    ),
                    "missing_response_options": _json_safe(
                        response_metadata.get("missing_response_options", [])
                    ),
                }
                source_details.append(detail)

                substantive_options = detail["response_options"]
                if not substantive_options:
                    common_response_options = None
                    continue

                if common_response_options is None:
                    common_response_options = substantive_options
                elif common_response_options != substantive_options:
                    common_response_options = []

            provenance_map[output_column] = {
                "role": str(row.get("role", "") or "").strip(),
                "category": str(row.get("category", "") or "").strip(),
                "source_variables": source_variables,
                "source_labels": source_labels,
                "source_details": source_details,
                "transformation": transformation,
                "selection_rationale": selection_rationale,
                "construction_type": self._classify_feature_construction(
                    transformation,
                    source_labels,
                ),
                "response_options": common_response_options or [],
                "missing_value_handling": self._extract_missing_value_handling(
                    transformation,
                ),
                "source_count": max(len(source_variables), len(source_labels)),
                "dictionary_path": str(dictionary_path),
            }

        self._feature_dictionary_cache[cache_key] = provenance_map
        return provenance_map

    def _find_feature_dictionary_path_by_feature_names(
        self,
        feature_names: List[str],
    ) -> Optional[Path]:
        normalized_feature_names = {
            str(feature_name).strip()
            for feature_name in feature_names
            if str(feature_name).strip()
        }
        if not normalized_feature_names:
            return None

        repo_root = Path(__file__).resolve().parents[2]
        search_roots = [repo_root, Path.home() / "Downloads"]
        candidate_matches: List[Tuple[int, int, Path]] = []
        seen_roots: set[str] = set()

        for root in search_roots:
            try:
                resolved_root = root.resolve()
            except Exception:
                continue
            if not resolved_root.exists() or not resolved_root.is_dir():
                continue
            root_key = str(resolved_root)
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)

            try:
                dictionary_paths = list(resolved_root.rglob("*_feature_dictionary.csv"))
            except Exception:
                continue

            for dictionary_path in dictionary_paths:
                try:
                    dictionary_df = pd.read_csv(
                        dictionary_path,
                        dtype=str,
                        encoding="utf-8-sig",
                        usecols=["output_column"],
                    )
                except Exception:
                    continue

                output_columns = {
                    str(value).strip()
                    for value in dictionary_df["output_column"].dropna().tolist()
                    if str(value).strip()
                }
                if not normalized_feature_names.issubset(output_columns):
                    continue

                candidate_matches.append(
                    (
                        len(output_columns),
                        len(dictionary_path.parts),
                        dictionary_path,
                    )
                )

        if not candidate_matches:
            return None

        candidate_matches.sort(key=lambda item: (item[0], item[1], str(item[2])))
        return candidate_matches[0][2]

    def _build_feature_provenance_map(
        self,
        dataset_name: Optional[str],
        dataset_path: Optional[Path],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        dictionary_path = self._find_feature_dictionary_path(dataset_name, dataset_path)
        if dictionary_path is None and feature_names:
            dictionary_path = self._find_feature_dictionary_path_by_feature_names(
                feature_names,
            )

        if dictionary_path is None:
            cache_key = f"missing::{dataset_name or ''}::{','.join(feature_names or [])}"
            cached_missing = self._feature_dictionary_cache.get(cache_key)
            if cached_missing is not None:
                return cached_missing
            self._feature_dictionary_cache[cache_key] = {}
            return {}

        return self._load_feature_provenance_from_path(dictionary_path)

    @staticmethod
    def _parse_value_label_options(raw_value_labels: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_parts = MLService._split_dictionary_values(raw_value_labels)
        substantive_options: List[Dict[str, Any]] = []
        missing_options: List[Dict[str, Any]] = []

        for part in raw_parts:
            if "=" in part:
                raw_value, raw_label = part.split("=", 1)
                value = raw_value.strip()
                label = raw_label.strip()
            else:
                value = ""
                label = part.strip()

            option = {
                "value": value,
                "label": label,
            }
            if MLService._is_missing_response_option(value, label):
                missing_options.append(option)
            else:
                substantive_options.append(option)

        return substantive_options, missing_options

    @staticmethod
    def _is_missing_response_option(value: str, label: str) -> bool:
        normalized_value = str(value).strip()
        normalized_label = str(label).strip().lower()

        try:
            numeric_value = float(normalized_value)
        except Exception:
            numeric_value = None

        if numeric_value is not None and numeric_value < 0:
            return True

        missing_markers = [
            "auslassen",
            "unklare beantwortung",
            "fragebogenrotation",
            "kein fragebogen",
            "fehlend",
            "missing",
            "nicht kalkulierbar",
            "ungueltig",
        ]
        return any(marker in normalized_label for marker in missing_markers)

    def _load_column_mapping_from_path(
        self,
        column_mapping_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        cache_key = str(column_mapping_path)
        cached = self._column_mapping_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            mapping_df = pd.read_csv(
                column_mapping_path,
                dtype=str,
                encoding="utf-8-sig",
            )
        except Exception:
            self._column_mapping_cache[cache_key] = {}
            return {}

        required_columns = {
            "original_spss_variable",
            "description",
            "value_labels",
        }
        if not required_columns.issubset(set(mapping_df.columns)):
            self._column_mapping_cache[cache_key] = {}
            return {}

        mapping: Dict[str, Dict[str, Any]] = {}
        for _, row in mapping_df.iterrows():
            variable_name = str(row.get("original_spss_variable", "") or "").strip()
            if not variable_name:
                continue

            response_options, missing_response_options = self._parse_value_label_options(
                row.get("value_labels"),
            )
            mapping[variable_name] = {
                "description": str(row.get("description", "") or "").strip(),
                "response_options": response_options,
                "missing_response_options": missing_response_options,
            }

        self._column_mapping_cache[cache_key] = mapping
        return mapping

    def _find_readable_column_mapping_path(
        self,
        dictionary_path: Path,
    ) -> Optional[Path]:
        direct_candidates = sorted(
            dictionary_path.parent.glob("*_readable_v1_column_mapping.csv")
        )
        if direct_candidates:
            return direct_candidates[0]

        repo_root = Path(__file__).resolve().parents[2]
        search_roots = [repo_root, Path.home() / "Downloads"]
        seen: set[str] = set()

        for root in search_roots:
            try:
                resolved_root = root.resolve()
            except Exception:
                continue
            if not resolved_root.exists() or not resolved_root.is_dir():
                continue
            root_key = str(resolved_root)
            if root_key in seen:
                continue
            seen.add(root_key)
            try:
                matches = list(
                    resolved_root.rglob("*_readable_v1_column_mapping.csv")
                )
            except Exception:
                continue
            if matches:
                matches.sort(key=lambda path: len(path.parts))
                return matches[0]

        return None

    def _build_response_option_map_for_dictionary(
        self,
        dictionary_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        column_mapping_path = self._find_readable_column_mapping_path(dictionary_path)
        if column_mapping_path is None:
            return {}
        return self._load_column_mapping_from_path(column_mapping_path)

    def _find_feature_dictionary_path(
        self,
        dataset_name: Optional[str],
        dataset_path: Optional[Path],
    ) -> Optional[Path]:
        normalized_dataset_name = str(dataset_name or "").strip()
        if not normalized_dataset_name:
            return None

        expected_name = (
            f"{normalized_dataset_name[:-4]}_feature_dictionary.csv"
            if normalized_dataset_name.lower().endswith(".csv")
            else f"{normalized_dataset_name}_feature_dictionary.csv"
        )

        direct_candidates: List[Path] = []
        if dataset_path is not None:
            direct_candidates.append(dataset_path.parent / expected_name)
        direct_candidates.append(self.datasets_dir / expected_name)

        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate

        repo_root = Path(__file__).resolve().parents[2]
        search_roots = [repo_root, Path.home() / "Downloads"]
        seen: set[str] = set()

        for root in search_roots:
            try:
                resolved_root = root.resolve()
            except Exception:
                continue
            if not resolved_root.exists() or not resolved_root.is_dir():
                continue
            root_key = str(resolved_root)
            if root_key in seen:
                continue
            seen.add(root_key)
            try:
                matches = list(resolved_root.rglob(expected_name))
            except Exception:
                continue
            if matches:
                matches.sort(key=lambda path: len(path.parts))
                return matches[0]

        return None

    def _enrich_feature_schema_with_provenance(
        self,
        feature_schema: List[Dict[str, Any]],
        *,
        dataset_name: Optional[str],
        dataset_path: Optional[Path],
    ) -> List[Dict[str, Any]]:
        provenance_map = self._build_feature_provenance_map(
            dataset_name,
            dataset_path,
            [str(item.get("name", "") or "").strip() for item in feature_schema],
        )
        if not provenance_map:
            return feature_schema

        enriched_schema: List[Dict[str, Any]] = []
        for item in feature_schema:
            enriched_item = dict(item)
            feature_name = str(enriched_item.get("name", "") or "").strip()
            provenance = provenance_map.get(feature_name)
            if provenance:
                enriched_item["feature_provenance"] = provenance
                numeric_domain = self._infer_numeric_domain_from_provenance(
                    enriched_item,
                    provenance,
                )
                if numeric_domain is not None:
                    domain_min, domain_max = numeric_domain
                    enriched_item["min_value"] = domain_min
                    enriched_item["max_value"] = domain_max
                    default_value = enriched_item.get("default_value")
                    if default_value is not None:
                        try:
                            default_numeric = float(default_value)
                        except (TypeError, ValueError):
                            default_numeric = None
                        if default_numeric is not None:
                            enriched_item["default_value"] = round(
                                min(max(default_numeric, domain_min), domain_max),
                                2,
                            )
            enriched_schema.append(enriched_item)
        return enriched_schema

    @staticmethod
    def _infer_numeric_domain_from_provenance(
        feature_schema_entry: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        if str(feature_schema_entry.get("feature_type")) != "numeric":
            return None
        if str(provenance.get("construction_type")) != "item_mean":
            return None

        response_options = provenance.get("response_options") or []
        numeric_values: List[float] = []
        for option in response_options:
            if not isinstance(option, dict):
                continue
            try:
                numeric_value = float(option.get("value"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_value):
                continue
            numeric_values.append(numeric_value)

        if len(numeric_values) < 2:
            return None

        return (float(min(numeric_values)), float(max(numeric_values)))

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
        series = pd.to_numeric(self._get_public_training_series(feature_name), errors="coerce").dropna()
        if series.empty or not self._is_integer_like_numeric(series):
            return False

        unique_count = int(series.nunique(dropna=True))
        return unique_count <= 13

    def _feature_can_be_numeric(self, feature_name: str) -> bool:
        """Whether a feature can safely be treated as numeric for charts."""
        if feature_name in self.num_features:
            return True
        series = self._get_public_training_series(feature_name)
        non_null = series.dropna()
        if non_null.empty:
            return False

        numeric = pd.to_numeric(non_null, errors="coerce")
        return bool(numeric.notna().all())

    @classmethod
    def _sanitize_chart_label_map(cls, raw_labels: Any) -> Dict[str, str]:
        if not isinstance(raw_labels, dict):
            return {}
        sanitized: Dict[str, str] = {}
        for raw_key, raw_label in raw_labels.items():
            label_str = str(raw_label).strip()
            if not label_str:
                continue
            sanitized[cls._stringify_chart_value(raw_key)] = label_str
        return sanitized

    @classmethod
    def _normalize_stored_feature_chart_setting(
        cls,
        raw_setting: Any,
    ) -> Dict[str, Any]:
        if not isinstance(raw_setting, dict):
            return {}
        categorical_value_labels = cls._sanitize_chart_label_map(
            raw_setting.get("categorical_value_labels", raw_setting.get("value_labels")),
        )
        numeric_tick_labels = cls._sanitize_chart_label_map(
            raw_setting.get("numeric_tick_labels"),
        )
        return {
            "treat_as_categorical": bool(raw_setting.get("treat_as_categorical")),
            "treat_as_numeric": bool(raw_setting.get("treat_as_numeric")),
            "categorical_value_labels": categorical_value_labels,
            "numeric_tick_labels": numeric_tick_labels,
        }

    def _get_feature_chart_setting(self, feature_name: str) -> Dict[str, Any]:
        normalized = self._normalize_stored_feature_chart_setting(
            self.feature_chart_settings.get(feature_name, {}),
        )
        if not normalized:
            return {
                "treat_as_categorical": False,
                "treat_as_numeric": False,
                "categorical_value_labels": {},
                "numeric_tick_labels": {},
            }
        return normalized

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

    def _get_numeric_chart_x_values(self, feature_name: str) -> List[float]:
        if feature_name in self.cat_features:
            return [
                float(numeric_x)
                for _, numeric_x in self._get_numeric_chart_values_for_categorical_feature(
                    feature_name,
                )
            ]

        public_numeric_series = pd.to_numeric(
            self._get_public_training_series(feature_name),
            errors="coerce",
        ).dropna()
        if public_numeric_series.empty:
            return []

        schema = self.feature_schema_map.get(feature_name, {}) or {}
        schema_min = schema.get("min_value")
        schema_max = schema.get("max_value")
        try:
            min_val = float(schema_min)
        except (TypeError, ValueError):
            min_val = float(public_numeric_series.min())
        try:
            max_val = float(schema_max)
        except (TypeError, ValueError):
            max_val = float(public_numeric_series.max())

        if not np.isfinite(min_val):
            min_val = float(public_numeric_series.min())
        if not np.isfinite(max_val):
            max_val = float(public_numeric_series.max())
        if max_val < min_val:
            min_val = float(public_numeric_series.min())
            max_val = float(public_numeric_series.max())

        if min_val == max_val:
            return [min_val]
        return np.linspace(min_val, max_val, 30).tolist()

    def _get_numeric_chart_label_values(self, feature_name: str) -> List[str]:
        numeric_x_values = self._get_numeric_chart_x_values(feature_name)
        if not numeric_x_values:
            return []

        min_val = min(numeric_x_values)
        max_val = max(numeric_x_values)
        start = int(np.ceil(min_val - 1e-9))
        end = int(np.floor(max_val + 1e-9))
        if end < start:
            return []
        return [str(value) for value in range(start, end + 1)]

    def _build_numeric_chart_tick_labels(
        self,
        x_values: List[float],
        numeric_tick_labels: Dict[str, str],
    ) -> Optional[List[str]]:
        labels: List[str] = []
        has_custom_labels = False
        for x_value in x_values:
            label = str(
                numeric_tick_labels.get(self._stringify_chart_value(x_value), "")
            ).strip()
            if label:
                has_custom_labels = True
            labels.append(label)
        return labels if has_custom_labels else None

    def get_chart_display_settings(self) -> Dict[str, Any]:
        return {
            "show_missing_bars": bool(self.show_missing_bars),
        }

    def update_chart_display_settings(self, show_missing_bars: bool) -> Dict[str, Any]:
        self.show_missing_bars = bool(show_missing_bars)
        # Clear cached shape-function snapshots so the next fetch reflects the new mode.
        self.original_shape_functions = {}
        self.shape_function_offsets = {}
        self._persist_active_dataset_metadata()
        return self.get_chart_display_settings()

    def _get_chart_x_values(
        self,
        feature_name: str,
        *,
        treat_as_categorical: Optional[bool] = None,
    ) -> List[str]:
        """Get raw x-axis values that should be used for a categorical chart."""
        series = self._get_public_training_series(feature_name)
        if series.empty and feature_name not in self.feature_schema_map:
            return []

        if treat_as_categorical is None:
            treat_as_categorical = self._is_feature_categorical_for_chart(feature_name)

        if not treat_as_categorical:
            return []

        if str(self.feature_schema_map.get(feature_name, {}).get("feature_type")) == "categorical":
            options = self.feature_schema_map.get(feature_name, {}).get("categorical_options", [])
            base_values = [self._stringify_chart_value(v) for v in options]
        else:
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            uniques = sorted(numeric_series.unique().tolist())
            base_values = [self._stringify_chart_value(v) for v in uniques]
            if self.show_missing_bars and self._get_public_missing_count(feature_name) > 0:
                base_values.append(MISSING_CATEGORY_VALUE)

        deduped: List[str] = []
        seen = set()
        for raw in base_values:
            if not self.show_missing_bars and raw == MISSING_CATEGORY_VALUE:
                continue
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
        if feature_name not in self._public_feature_names():
            raise ValueError(f"Unknown feature: {feature_name}")

        base_feature_type = str(
            self.feature_schema_map.get(feature_name, {}).get("feature_type", "numeric")
        )
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
        allowed_categorical_values = set(
            self._get_chart_x_values(feature_name, treat_as_categorical=True)
        )
        available_categorical_values = (
            self._get_chart_x_values(feature_name, treat_as_categorical=True)
            if can_be_categorical or base_feature_type == "categorical"
            else []
        )
        available_numeric_values = (
            self._get_numeric_chart_label_values(feature_name)
            if can_be_numeric or base_feature_type == "numeric"
            else []
        )
        numeric_domain_values = (
            self._get_numeric_chart_x_values(feature_name)
            if can_be_numeric or base_feature_type == "numeric"
            else []
        )
        allowed_numeric_values = set(available_numeric_values)
        categorical_value_labels = {
            str(raw): str(label).strip()
            for raw, label in (stored.get("categorical_value_labels", {}) or {}).items()
            if str(label).strip()
            and (not allowed_categorical_values or str(raw) in allowed_categorical_values)
        }
        numeric_tick_labels = {
            str(raw): str(label).strip()
            for raw, label in (stored.get("numeric_tick_labels", {}) or {}).items()
            if str(label).strip()
            and (not allowed_numeric_values or str(raw) in allowed_numeric_values)
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
            "categorical_value_labels": categorical_value_labels,
            "numeric_tick_labels": numeric_tick_labels,
            "value_labels": categorical_value_labels,
            "available_categorical_values": available_categorical_values,
            "available_numeric_values": available_numeric_values,
            "numeric_domain_min": (
                float(min(numeric_domain_values)) if numeric_domain_values else None
            ),
            "numeric_domain_max": (
                float(max(numeric_domain_values)) if numeric_domain_values else None
            ),
        }

    def update_feature_chart_setting(
        self,
        feature_name: str,
        treat_as_categorical: bool,
        treat_as_numeric: bool = False,
        categorical_value_labels: Optional[Dict[str, str]] = None,
        numeric_tick_labels: Optional[Dict[str, str]] = None,
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
        allowed_categorical_values = set(
            self._get_chart_x_values(feature_name, treat_as_categorical=True)
        )
        allowed_numeric_values = {
            self._stringify_chart_value(value)
            for value in self._get_numeric_chart_label_values(feature_name)
        }

        source_categorical_labels = (
            current.get("categorical_value_labels", {})
            if categorical_value_labels is None and value_labels is None
            else (
                categorical_value_labels
                if categorical_value_labels is not None
                else (value_labels or {})
            )
        )
        sanitized_categorical_labels: Dict[str, str] = {}
        for raw_key, raw_label in (source_categorical_labels or {}).items():
            key = self._stringify_chart_value(raw_key)
            label = str(raw_label).strip()
            if not label:
                continue
            if allowed_categorical_values and key not in allowed_categorical_values:
                continue
            sanitized_categorical_labels[key] = label

        source_numeric_labels = (
            current.get("numeric_tick_labels", {})
            if numeric_tick_labels is None
            else (numeric_tick_labels or {})
        )
        sanitized_numeric_labels: Dict[str, str] = {}
        for raw_key, raw_label in (source_numeric_labels or {}).items():
            key = self._stringify_chart_value(raw_key)
            label = str(raw_label).strip()
            if not label:
                continue
            if allowed_numeric_values and key not in allowed_numeric_values:
                continue
            sanitized_numeric_labels[key] = label

        normalized_setting = {
            "treat_as_categorical": normalized_treat_as_categorical,
            "treat_as_numeric": normalized_treat_as_numeric,
            "categorical_value_labels": sanitized_categorical_labels,
            "numeric_tick_labels": sanitized_numeric_labels,
        }

        if (
            not normalized_setting["treat_as_categorical"]
            and not normalized_setting["treat_as_numeric"]
            and not normalized_setting["categorical_value_labels"]
            and not normalized_setting["numeric_tick_labels"]
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

    def _build_public_feature_schema(
        self,
        selected_features: List[str],
        public_numeric_features: List[str],
        public_frame: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Build the public-facing schema from the original selected features only."""
        schema: List[Dict[str, Any]] = []
        public_numeric_set = set(public_numeric_features)

        for feature_name in selected_features:
            if feature_name in public_numeric_set:
                numeric_series = pd.to_numeric(public_frame[feature_name], errors="coerce")
                valid_numeric_series = numeric_series.dropna()

                if valid_numeric_series.empty:
                    min_val = 0.0
                    max_val = 0.0
                    default_val = 0.0
                else:
                    min_val = float(valid_numeric_series.min())
                    max_val = float(valid_numeric_series.max())
                    default_val = float(valid_numeric_series.median())

                schema.append(
                    {
                        "name": feature_name,
                        "feature_type": "numeric",
                        "default_value": round(float(default_val), 2),
                        "min_value": min_val,
                        "max_value": max_val,
                    }
                )
                continue

            cat_series = (
                public_frame[feature_name]
                .astype(object)
                .where(public_frame[feature_name].notna(), MISSING_CATEGORY_VALUE)
                .astype(str)
            )
            unique_values = [str(v) for v in pd.unique(cat_series)]
            unique_values = sorted(unique_values)
            if not unique_values:
                unique_values = [MISSING_CATEGORY_VALUE]

            observed_counts = cat_series[cat_series != MISSING_CATEGORY_VALUE].value_counts()
            if not observed_counts.empty:
                default_val = str(observed_counts.index[0])
            else:
                default_val = unique_values[0]

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
            "show_missing_bars": bool(self.show_missing_bars),
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
            "feature_chart_settings": {
                str(feature_name): normalized
                for feature_name, raw_setting in (
                    _json_safe(artifact.get("feature_chart_settings") or {})
                ).items()
                for normalized in [self._normalize_stored_feature_chart_setting(raw_setting)]
                if normalized
            },
            "show_missing_bars": bool(artifact.get("show_missing_bars", False)),
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
        self.missing_indicator_map = {}
        self.numeric_missing_placeholder_values = {}
        self.feature_chart_settings = dict(validated["feature_chart_settings"])
        self.show_missing_bars = bool(validated.get("show_missing_bars", False))
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
            "features": self.selected_feature_columns if self.selected_feature_columns else [],
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
            "show_missing_bars": bool(self.show_missing_bars),
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
        self.missing_indicator_map = {}
        self.numeric_missing_placeholder_values = {}
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
            "missing_indicator_map": dict(self.missing_indicator_map),
            "numeric_missing_placeholder_values": dict(
                self.numeric_missing_placeholder_values
            ),
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
            feature_processing_metadata,
        ) = prepare_training_data(
            csv_path=str(resolved_path),
            target_column=target_column,
            feature_columns=feature_columns,
        )

        public_numeric_features = list(
            feature_processing_metadata.get("public_numeric_features", [])
        )
        missing_indicator_map = dict(
            feature_processing_metadata.get("missing_indicator_map", {})
        )
        numeric_missing_placeholder_values = {
            str(key): float(value)
            for key, value in (
                feature_processing_metadata.get(
                    "numeric_missing_placeholder_values", {}
                )
                or {}
            ).items()
        }
        public_frame = df_loaded.loc[:, list(selected_features)].copy()
        rebuilt_feature_schema = self._build_public_feature_schema(
            list(selected_features),
            public_numeric_features,
            public_frame,
        )
        effective_dataset_name = (
            Path(str(dataset_name)).name
            if dataset_name
            else (
                self.active_dataset_name
                or (Path(resolved_path_str).name if resolved_path_str else None)
            )
        )
        rebuilt_feature_schema = self._enrich_feature_schema_with_provenance(
            rebuilt_feature_schema,
            dataset_name=effective_dataset_name,
            dataset_path=resolved_path,
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
                if feature in self.selected_feature_columns
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
        self.missing_indicator_map = missing_indicator_map
        self.numeric_missing_placeholder_values = numeric_missing_placeholder_values
        self.feature_chart_settings = {
            feature: setting
            for feature, setting in self.feature_chart_settings.items()
            if feature in self.selected_feature_columns
        }

        self.active_dataset_id = resolved_dataset_id
        self.active_dataset_path = resolved_path_str
        self.active_dataset_name = (
            effective_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None)
        )
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
            "features": self.selected_feature_columns,
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
        if (
            len(feature_columns) != len(self.selected_feature_columns)
            or set(feature_columns) != set(self.selected_feature_columns)
        ):
            raise ValueError(
                "Comparison dataset feature columns must match the primary dataset selected feature columns exactly"
            )

    def _normalize_comparison_feature_order(
        self,
        feature_columns: List[str],
    ) -> List[str]:
        primary_feature_set = set(self.selected_feature_columns)
        normalized = [
            feature_name
            for feature_name in self.selected_feature_columns
            if feature_name in primary_feature_set and feature_name in set(feature_columns)
        ]
        if len(normalized) != len(feature_columns):
            return list(feature_columns)
        return normalized

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
            feature_processing_metadata,
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
        selected_features = self._normalize_comparison_feature_order(
            selected_features,
        )

        public_numeric_features = list(
            feature_processing_metadata.get("public_numeric_features", [])
        )
        public_numeric_features = [
            feature_name
            for feature_name in selected_features
            if feature_name in set(public_numeric_features)
        ]
        comparison_missing_indicator_map = dict(
            feature_processing_metadata.get("missing_indicator_map", {})
        )
        comparison_numeric_missing_placeholder_values = {
            str(key): float(value)
            for key, value in (
                feature_processing_metadata.get(
                    "numeric_missing_placeholder_values", {}
                )
                or {}
            ).items()
        }
        public_frame = df_loaded.loc[:, list(selected_features)].copy()
        rebuilt_feature_schema = self._build_public_feature_schema(
            list(selected_features),
            public_numeric_features,
            public_frame,
        )
        comparison_dataset_name = (
            Path(str(dataset_name)).name
            if dataset_name
            else Path(str(resolved_path)).name
        )
        rebuilt_feature_schema = self._enrich_feature_schema_with_provenance(
            rebuilt_feature_schema,
            dataset_name=comparison_dataset_name,
            dataset_path=resolved_path,
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
        self.comparison_missing_indicator_map = comparison_missing_indicator_map
        self.comparison_numeric_missing_placeholder_values = (
            comparison_numeric_missing_placeholder_values
        )
        self.comparison_dataset_id = resolved_dataset_id
        self.comparison_dataset_path = str(resolved_path)
        self.comparison_dataset_name = comparison_dataset_name

        return {
            "total_records": len(df_loaded),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": selected_features,
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

    def _public_feature_names(self) -> List[str]:
        return (
            list(self.selected_feature_columns)
            if self.selected_feature_columns
            else list(self.feature_schema_map.keys())
        )

    def _missing_indicator_feature_name(self, feature_name: str) -> Optional[str]:
        return self.missing_indicator_map.get(feature_name)

    def _is_missing_category_value(self, value: Any) -> bool:
        return str(value) == MISSING_CATEGORY_VALUE

    def _is_missing_numeric_feature_in_row(
        self,
        feature_name: str,
        row: Any,
    ) -> bool:
        indicator_name = self._missing_indicator_feature_name(feature_name)
        if not indicator_name:
            return False
        try:
            indicator_value = row[indicator_name]
        except Exception:
            try:
                indicator_value = row.get(indicator_name)
            except Exception:
                indicator_value = None
        return self._is_missing_category_value(indicator_value)

    def _get_public_training_series(self, feature_name: str) -> pd.Series:
        if (
            self.df is None
            or self.X_train is None
            or feature_name not in self.selected_feature_columns
            or feature_name not in self.df.columns
        ):
            return pd.Series(dtype=object)
        return self.df.loc[self.X_train.index, feature_name]

    def _get_public_missing_count(self, feature_name: str) -> int:
        raw_series = self._get_public_training_series(feature_name)
        if raw_series.empty:
            return 0
        if feature_name in self.num_features:
            return int(pd.to_numeric(raw_series, errors="coerce").isna().sum())
        return int(raw_series.isna().sum())

    def _normalize_public_series_for_categorical_chart(
        self,
        feature_name: str,
    ) -> pd.Series:
        raw_series = self._get_public_training_series(feature_name)
        if raw_series.empty:
            return pd.Series(dtype=object)
        if feature_name in self.num_features:
            numeric = pd.to_numeric(raw_series, errors="coerce")
            return numeric.map(
                lambda value: (
                    MISSING_CATEGORY_VALUE
                    if pd.isna(value)
                    else self._stringify_chart_value(value)
                )
            )
        return (
            raw_series.astype(object)
            .where(raw_series.notna(), MISSING_CATEGORY_VALUE)
            .astype(str)
        )

    def _predict_shape_function_sample(self, sample_data: Dict[str, Any]) -> float:
        sample = pd.DataFrame([sample_data])[self.feature_names]

        for cat_feat in self.cat_features:
            sample[cat_feat] = sample[cat_feat].astype(str)
        for num_feat in self.num_features:
            sample[num_feat] = sample[num_feat].astype(float)

        pred = self.model.predict(sample)
        return float(pred[0]) if hasattr(pred, "__iter__") else float(pred)

    def _build_numeric_chart_missing_bucket(
        self,
        feature_name: str,
        baseline: Dict[str, Any],
        centered_mean: float,
    ) -> Optional[Dict[str, Any]]:
        missing_count = self._get_public_missing_count(feature_name)
        if missing_count <= 0:
            return None

        sample_data = baseline.copy()
        if feature_name in self.num_features:
            indicator_name = self._missing_indicator_feature_name(feature_name)
            if not indicator_name:
                return None
            sample_data[feature_name] = float(
                self.numeric_missing_placeholder_values.get(
                    feature_name,
                    self.feature_schema_map.get(feature_name, {}).get(
                        "default_value",
                        0.0,
                    ),
                )
            )
            sample_data[indicator_name] = MISSING_CATEGORY_VALUE
        elif feature_name in self.cat_features:
            sample_data[feature_name] = MISSING_CATEGORY_VALUE
        else:
            return None

        prediction = self._predict_shape_function_sample(sample_data)
        return {
            "label": MISSING_CATEGORY_VALUE,
            "count": missing_count,
            "y_value": prediction - centered_mean,
        }

    def _build_model_row_from_public_features(
        self,
        normalized_public_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for feature_name in self._public_feature_names():
            schema = self.feature_schema_map.get(feature_name, {})
            feature_type = schema.get("feature_type")
            value = normalized_public_features.get(feature_name)

            if feature_type == "numeric":
                indicator_name = self._missing_indicator_feature_name(feature_name)
                if value is None:
                    row[feature_name] = float(
                        self.numeric_missing_placeholder_values.get(feature_name, 0.0)
                    )
                    if indicator_name:
                        row[indicator_name] = MISSING_CATEGORY_VALUE
                else:
                    row[feature_name] = float(value)
                    if indicator_name:
                        row[indicator_name] = "Observed"
            else:
                row[feature_name] = (
                    MISSING_CATEGORY_VALUE if value is None else str(value)
                )

        return row

    def _validate_and_normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction payload against loaded feature schema."""
        if not self.feature_schema_map:
            raise ValueError("Feature schema is not available yet")

        provided_keys = set(features.keys())
        expected_keys = set(self._public_feature_names())
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
        for feature_name in self._public_feature_names():
            value = features[feature_name]
            schema = self.feature_schema_map.get(feature_name, {})
            feature_type = schema.get("feature_type")

            if feature_type == "numeric":
                if value is None or (isinstance(value, str) and not value.strip()):
                    normalized[feature_name] = None
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature '{feature_name}' must be numeric") from exc
                if not np.isfinite(numeric_value):
                    normalized[feature_name] = None
                    continue
                normalized[feature_name] = numeric_value
            else:
                if value is None or (isinstance(value, str) and not value.strip()):
                    normalized[feature_name] = None
                    continue
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
        input_df = pd.DataFrame([self._build_model_row_from_public_features(normalized)])
        for cat_feature in self.cat_features:
            if cat_feature in input_df.columns:
                input_df[cat_feature] = input_df[cat_feature].astype(str)
        for num_feature in self.num_features:
            if num_feature in input_df.columns:
                input_df[num_feature] = pd.to_numeric(input_df[num_feature], errors="coerce")
        return input_df

    def _get_offset_for_row(self, feature_name: str, row: Any) -> float:
        """Get a feature offset for one row, skipping numeric offsets on missing values."""
        if self._is_missing_numeric_feature_in_row(feature_name, row):
            return 0.0
        try:
            value = row[feature_name]
        except Exception:
            try:
                value = row.get(feature_name)
            except Exception:
                return 0.0
        return self._get_offset_for_value(feature_name, value)

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
                total_offset += self._get_offset_for_row(feature_name, input_df.iloc[0])

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
                total_offset += self._get_offset_for_row(feature_name, input_df.iloc[0])

        self.shape_function_offsets = original_offsets
        return base_prediction + total_offset

    def batch_predict(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Make batch predictions."""
        return [self.predict(features) for features in features_list]

    # ==================== Shape functions ====================

    def get_shape_functions(self, include_distribution: bool = True) -> List[Dict[str, Any]]:
        """Get shape function data for all features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        if self.model_source == "imported" and self.original_shape_functions:
            return list(self.original_shape_functions.values())

        shape_functions = []
        for idx, feature_name in enumerate(self._public_feature_names()):
            try:
                shape_data = self._extract_shape_function(
                    feature_name,
                    idx,
                    include_distribution=include_distribution,
                )
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
            self.missing_indicator_map = dict(self.comparison_missing_indicator_map)
            self.numeric_missing_placeholder_values = dict(
                self.comparison_numeric_missing_placeholder_values
            )
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

            shape_functions = self.get_shape_functions(include_distribution=False)
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
            numeric_series = pd.to_numeric(
                self._get_public_training_series(num_feat),
                errors="coerce",
            ).dropna()
            if numeric_series.empty:
                baseline[num_feat] = float(
                    self.feature_schema_map.get(num_feat, {}).get("default_value", 0.0)
                )
            else:
                baseline[num_feat] = float(numeric_series.mean())
        for cat_feat in self.cat_features:
            mode = self.X_train[cat_feat].astype(str).mode()
            baseline[cat_feat] = str(mode.iloc[0]) if not mode.empty else "Unknown"
        return baseline

    def _build_numeric_distribution_bins(
        self,
        values: pd.Series,
        x_values: List[Any],
        missing_count: int = 0,
    ) -> Optional[Dict[str, Any]]:
        numeric_values = pd.to_numeric(values, errors="coerce").dropna()
        if numeric_values.empty:
            return None

        displayed_domain = [
            float(value)
            for value in x_values
            if value is not None and np.isfinite(pd.to_numeric(value, errors="coerce"))
        ]
        if displayed_domain:
            domain_min = float(min(displayed_domain))
            domain_max = float(max(displayed_domain))
        else:
            domain_min = float(numeric_values.min())
            domain_max = float(numeric_values.max())

        unique_count = int(numeric_values.nunique(dropna=True))
        if unique_count <= 1 or abs(domain_max - domain_min) < 1e-12:
            center = float(numeric_values.iloc[0])
            half_width = 0.5
            return {
                "chart_type": "numeric",
                "total_count": int(len(numeric_values) + missing_count),
                "bin_count": 1,
                "bins": [
                    {
                        "x0": float(center - half_width),
                        "x1": float(center + half_width),
                        "count": int(len(numeric_values)),
                        "center": center,
                    }
                ],
                "counts": [],
                "missing_count": int(missing_count),
                "missing_label": (
                    MISSING_CATEGORY_VALUE if missing_count > 0 else None
                ),
            }

        edges = np.linspace(
            domain_min,
            domain_max,
            SHAPE_DISTRIBUTION_DEFAULT_BIN_COUNT + 1,
        )
        counts, _ = np.histogram(numeric_values.to_numpy(dtype=float), bins=edges)
        bins = []
        for idx, count in enumerate(counts.tolist()):
            x0 = float(edges[idx])
            x1 = float(edges[idx + 1])
            bins.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "count": int(count),
                    "center": float((x0 + x1) / 2),
                }
            )

        return {
            "chart_type": "numeric",
            "total_count": int(len(numeric_values) + missing_count),
            "bin_count": len(bins),
            "bins": bins,
            "counts": [],
            "missing_count": int(missing_count),
            "missing_label": (
                MISSING_CATEGORY_VALUE if missing_count > 0 else None
            ),
        }

    def _build_numeric_distribution_for_categorical_feature(
        self,
        feature_name: str,
    ) -> Optional[Dict[str, Any]]:
        numeric_pairs = self._get_numeric_chart_values_for_categorical_feature(feature_name)
        if not numeric_pairs:
            return None

        normalized_series = self._normalize_public_series_for_categorical_chart(
            feature_name,
        )
        counts_map = normalized_series.value_counts().to_dict()
        missing_count = int(counts_map.get(MISSING_CATEGORY_VALUE, 0))
        centers = [float(numeric_x) for _, numeric_x in numeric_pairs]
        if len(centers) == 1:
            edges = [centers[0] - 0.5, centers[0] + 0.5]
        else:
            edges = [centers[0] - ((centers[1] - centers[0]) / 2)]
            for idx in range(len(centers) - 1):
                edges.append((centers[idx] + centers[idx + 1]) / 2)
            edges.append(centers[-1] + ((centers[-1] - centers[-2]) / 2))

        bins = []
        total_count = 0
        for idx, (raw_value, numeric_x) in enumerate(numeric_pairs):
            count = int(counts_map.get(str(raw_value), 0))
            total_count += count
            bins.append(
                {
                    "x0": float(edges[idx]),
                    "x1": float(edges[idx + 1]),
                    "count": count,
                    "center": float(numeric_x),
                }
            )

        return {
            "chart_type": "numeric",
            "total_count": total_count + missing_count,
            "bin_count": len(bins),
            "bins": bins,
            "counts": [],
            "missing_count": missing_count,
            "missing_label": (
                MISSING_CATEGORY_VALUE if missing_count > 0 else None
            ),
        }

    def _build_categorical_distribution(
        self,
        feature_name: str,
        x_values: List[Any],
        x_tick_labels: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_series = self._normalize_public_series_for_categorical_chart(
            feature_name,
        )

        counts_map = normalized_series.value_counts().to_dict()
        ordered_counts = []
        total_count = 0
        labels = (
            list(x_tick_labels)
            if isinstance(x_tick_labels, list) and len(x_tick_labels) == len(x_values)
            else None
        )

        for idx, raw_value in enumerate(x_values):
            normalized_value = self._stringify_chart_value(raw_value)
            count = int(counts_map.get(normalized_value, 0))
            total_count += count
            ordered_counts.append(
                {
                    "x_value": raw_value,
                    "label": labels[idx] if labels is not None else normalized_value,
                    "count": count,
                }
            )

        return {
            "chart_type": "categorical",
            "total_count": total_count,
            "bin_count": len(ordered_counts),
            "bins": [],
            "counts": ordered_counts,
            "missing_count": int(counts_map.get(MISSING_CATEGORY_VALUE, 0)),
            "missing_label": MISSING_CATEGORY_VALUE,
        }

    def _build_shape_function_distribution(
        self,
        feature_name: str,
        feature_type: str,
        x_values: List[Any],
        x_tick_labels: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if feature_name not in self._public_feature_names():
            return None

        if feature_type == "categorical":
            return self._build_categorical_distribution(
                feature_name,
                x_values,
                x_tick_labels=x_tick_labels,
            )

        if feature_name in self.cat_features:
            return self._build_numeric_distribution_for_categorical_feature(
                feature_name,
            )

        return self._build_numeric_distribution_bins(
            self._get_public_training_series(feature_name),
            x_values,
            missing_count=self._get_public_missing_count(feature_name),
        )

    def _extract_shape_function(
        self,
        feature_name: str,
        feature_idx: int,
        include_distribution: bool = True,
    ) -> Dict[str, Any]:
        """Extract shape function data for a single feature."""
        chart_setting = self.get_feature_chart_setting(feature_name)
        is_categorical = chart_setting.get("chart_feature_type") == "categorical"
        categorical_value_labels = (
            chart_setting.get("categorical_value_labels", {}) or {}
        )
        numeric_tick_labels = chart_setting.get("numeric_tick_labels", {}) or {}
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
                    indicator_name = self._missing_indicator_feature_name(feature_name)
                    if raw_value == MISSING_CATEGORY_VALUE:
                        if not indicator_name:
                            continue
                        sample_data[feature_name] = float(
                            self.numeric_missing_placeholder_values.get(
                                feature_name,
                                self.feature_schema_map.get(
                                    feature_name,
                                    {},
                                ).get("default_value", 0.0),
                            )
                        )
                        sample_data[indicator_name] = MISSING_CATEGORY_VALUE
                    else:
                        try:
                            sample_data[feature_name] = float(raw_value)
                        except (TypeError, ValueError):
                            continue
                        if indicator_name:
                            sample_data[indicator_name] = "Observed"
                else:
                    sample_data[feature_name] = raw_value
                pred_val = self._predict_shape_function_sample(sample_data)
                shape_values.append(pred_val)
                effective_x_values.append(raw_value)

            if not effective_x_values:
                is_categorical = False

            if is_categorical:
                mean_val = np.mean(shape_values)
                shape_values = [value - mean_val for value in shape_values]
                x_tick_labels = [
                    str(categorical_value_labels.get(raw_value, raw_value))
                    for raw_value in effective_x_values
                ]

                result = {
                    "feature_name": feature_name,
                    "x_values": effective_x_values,
                    "x_tick_labels": x_tick_labels,
                    "y_values": shape_values,
                    "feature_type": "categorical",
                    "chart_config": chart_setting,
                }
                if include_distribution:
                    result["distribution"] = self._build_shape_function_distribution(
                        feature_name,
                        "categorical",
                        effective_x_values,
                        x_tick_labels=x_tick_labels,
                    )
                return result

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
                    pred_val = self._predict_shape_function_sample(sample_data)
                    shape_values.append(pred_val)
                    x_values_numeric.append(float(numeric_x))

                mean_val = np.mean(shape_values)
                shape_values = [value - mean_val for value in shape_values]
                missing_bucket = (
                    self._build_numeric_chart_missing_bucket(
                        feature_name,
                        baseline,
                        mean_val,
                    )
                    if self.show_missing_bars
                    else None
                )

                result = {
                    "feature_name": feature_name,
                    "x_values": x_values_numeric,
                    "y_values": shape_values,
                    "feature_type": "numeric",
                    "x_tick_labels": self._build_numeric_chart_tick_labels(
                        x_values_numeric,
                        numeric_tick_labels,
                    ),
                    "chart_config": chart_setting,
                    "missing_bucket": missing_bucket,
                }
                if include_distribution:
                    result["distribution"] = self._build_shape_function_distribution(
                        feature_name,
                        "numeric",
                        x_values_numeric,
                    )
                return result

        x_values_numeric = self._get_numeric_chart_x_values(feature_name)
        if not x_values_numeric:
            raise ValueError(f"No observed numeric values available for '{feature_name}'")

        shape_values = []
        for x_val in x_values_numeric:
            sample_data = baseline.copy()
            sample_data[feature_name] = float(x_val)
            indicator_name = self._missing_indicator_feature_name(feature_name)
            if indicator_name:
                sample_data[indicator_name] = "Observed"
            pred_val = self._predict_shape_function_sample(sample_data)
            shape_values.append(pred_val)

        mean_val = np.mean(shape_values)
        shape_values = [value - mean_val for value in shape_values]
        missing_bucket = (
            self._build_numeric_chart_missing_bucket(
                feature_name,
                baseline,
                mean_val,
            )
            if self.show_missing_bars
            else None
        )

        result = {
            "feature_name": feature_name,
            "x_values": x_values_numeric,
            "y_values": shape_values,
            "feature_type": "numeric",
            "x_tick_labels": self._build_numeric_chart_tick_labels(
                x_values_numeric,
                numeric_tick_labels,
            ),
            "chart_config": chart_setting,
            "missing_bucket": missing_bucket,
        }
        if include_distribution:
            result["distribution"] = self._build_shape_function_distribution(
                feature_name,
                "numeric",
                x_values_numeric,
            )
        return result

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
            "features": self.selected_feature_columns,
            "numeric_features": [
                feature["name"]
                for feature in self.feature_schema
                if feature.get("feature_type") == "numeric"
            ],
            "categorical_features": [
                feature["name"]
                for feature in self.feature_schema
                if feature.get("feature_type") == "categorical"
            ],
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
        numeric_features = [
            feature["name"]
            for feature in self.feature_schema
            if feature.get("feature_type") == "numeric"
        ]
        categorical_features = [
            feature["name"]
            for feature in self.feature_schema
            if feature.get("feature_type") == "categorical"
        ]

        for feature in numeric_features:
            values = pd.to_numeric(
                self._get_public_training_series(feature),
                errors="coerce",
            ).dropna().tolist()
            distributions[feature] = {
                "type": "numeric",
                "values": values[:1000],
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
            }

        for feature in categorical_features:
            raw_series = self._get_public_training_series(feature)
            value_counts = (
                raw_series.astype(object)
                .where(raw_series.notna(), MISSING_CATEGORY_VALUE)
                .astype(str)
                .value_counts()
                .to_dict()
            )
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
                for idx, (_, row) in enumerate(X.iterrows()):
                    offsets[idx] += self._get_offset_for_row(feature_name, row)

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
