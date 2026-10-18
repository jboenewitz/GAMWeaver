"""Machine Learning service for IGANN model training and prediction."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from igann import IGANN
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .data_processing import (
    MISSING_CATEGORY_VALUE,
    MISSING_INDICATOR_SUFFIX,
    prepare_training_data,
)
from .noop_edits import encode_numeric_storage_key

DATA_FILE_NAME = "bike.csv"
MAX_PREVIEW_ROWS = 500
SHAPE_DISTRIBUTION_DEFAULT_BIN_COUNT = 20
MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS = 100
MODEL_ARTIFACT_VERSION = "1.0"
MODEL_ARTIFACT_VERSION_WITH_EDITS = "1.1"
MODEL_ARTIFACT_VERSION_WITH_NOOP_EDITS = "1.2"
SUPPORTED_MODEL_ARTIFACT_VERSIONS = {
    MODEL_ARTIFACT_VERSION,
    MODEL_ARTIFACT_VERSION_WITH_EDITS,
    MODEL_ARTIFACT_VERSION_WITH_NOOP_EDITS,
}

# Grid profiles used by the legacy English export.  These are measurement-scale
# values, not model-dependent sample points: preserving them lets a re-exported
# model be compared to the older artifacts on the same x-axis.
LEGACY_NUMERIC_EXPORT_GRIDS: Dict[str, Tuple[float, ...]] = {
    "Student Experience – Social Integration": (
        1.0, 1.25, 4 / 3, 1.5, 5 / 3, 1.75, 2.0, 2.25, 7 / 3,
        2.5, 8 / 3, 2.75, 3.0, 3.25, 10 / 3, 3.5, 11 / 3, 3.75, 4.0,
    ),
    "German – Interest": (
        1.0, 1.25, 4 / 3, 1.5, 5 / 3, 1.75, 2.0, 2.25, 7 / 3,
        2.5, 8 / 3, 2.75, 3.0, 3.25, 10 / 3, 3.5, 11 / 3, 3.75, 4.0,
    ),
    "German – Anxiety": (
        1.0, 9 / 8, 8 / 7, 7 / 6, 5 / 4, 9 / 7, 4 / 3, 11 / 8,
        10 / 7, 3 / 2, 11 / 7, 13 / 8, 5 / 3, 12 / 7, 7 / 4, 9 / 5,
        11 / 6, 13 / 7, 15 / 8, 2.0, 17 / 8, 15 / 7, 13 / 6, 11 / 5,
        9 / 4, 16 / 7, 7 / 3, 19 / 8, 12 / 5, 17 / 7, 5 / 2, 18 / 7,
        13 / 5, 21 / 8, 8 / 3, 19 / 7, 11 / 4, 14 / 5, 17 / 6, 20 / 7,
        23 / 8, 3.0, 25 / 8, 22 / 7, 19 / 6, 13 / 4, 23 / 7, 10 / 3,
        27 / 8, 24 / 7, 7 / 2, 25 / 7, 29 / 8, 11 / 3, 26 / 7, 15 / 4,
        19 / 5, 23 / 6, 31 / 8, 4.0,
    ),
    "German – Perceived Performance Pressure": tuple(
        1.0 + index / 5 for index in range(16)
    ),
    "Digital Competencies – Computer and Tablet Competence": tuple(
        1.0 + index / 5 for index in range(16)
    ),
    "German Instruction – Supportive Learning Climate": tuple(
        1.0 + index / 7 for index in range(22)
    ),
}
DEFAULT_LEGACY_ONE_TO_FOUR_EXPORT_GRID: Tuple[float, ...] = tuple(
    1.0 + index / 5 for index in range(16)
)

# Public feature names from final_igann_model_v2_english_imputed_no_parentheses_anonymized.json.
# The source labels are the corresponding headers in the German case-study CSV.
CASE_STUDY_TARGET_COLUMN = "Zielvariable – Lesekompetenz Deutsch"
CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN: Dict[str, str] = {
    "Sozioökonomischer Status – Höchster ISEI-Wert der Eltern": (
        "Socioeconomic Status – Highest Parental ISEI Score"
    ),
    "Schüler:innenerleben – Soziale Integration (Skalenwert)": (
        "Student Experience – Social Integration"
    ),
    "Deutsch – Interesse (Skalenwert)": "German – Interest",
    "Deutsch – Angst (Skalenwert)": "German – Anxiety",
    "Deutsch – Wahrgenommener Leistungsdruck (Itemmittelwert 1–4)": (
        "German – Perceived Performance Pressure"
    ),
    "Digitale Kompetenzen – Computer- und Tabletkompetenz (Itemmittelwert)": (
        "Digital Competencies – Computer and Tablet Competence"
    ),
    "Deutschunterricht – Unterstützendes Lernklima (Itemmittelwert 1–4)": (
        "German Instruction – Supportive Learning Climate"
    ),
    "Lernschwierigkeit – Lese- und Rechtschreibschwierigkeiten (1 = ja)": (
        "Learning Difficulty – Reading and Spelling Difficulties"
    ),
    "Familiäre Lesekultur – Bücher zu Hause (1–6)": (
        "Family Reading Culture – Books at Home"
    ),
    "Sprachhintergrund – Zu Hause gesprochenes Deutsch (1 = nie, 3 = immer)": (
        "Language Background – German Spoken at Home"
    ),
}
CASE_STUDY_FEATURE_NAMES: Tuple[str, ...] = tuple(
    CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.values()
)


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


def _format_compare_chart_trace_user_name(user_name: Any) -> str:
    """Use the expert prefix for anonymized labels in comparison chart traces."""
    name = str(user_name or "").strip()
    match = re.fullmatch(r"P(\d+)", name, flags=re.IGNORECASE)
    return f"E{match.group(1)}" if match else name


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
        self._shape_functions: List[Dict[str, Any]] = []

        for shape_function in shape_functions:
            feature_name = str(shape_function.get("feature_name", "")).strip()
            if not feature_name:
                continue
            normalized_shape = _json_safe(shape_function)
            self._shape_by_feature[feature_name] = normalized_shape
            self._shape_functions.append(normalized_shape)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self._params)

    def export_shape_functions(self) -> List[Dict[str, Any]]:
        return [
            _json_safe(self._shape_by_feature[feature_name])
            for feature_name in self.feature_names
            if feature_name in self._shape_by_feature
        ]

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
        self.imported_training_feature_rows: Optional[List[Dict[str, Any]]] = None
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
        self.show_competence_levels = False
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
            "show_competence_levels": bool(self.show_competence_levels),
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
        show_competence_levels = data.get("show_competence_levels")
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
        if show_competence_levels is not None:
            self.show_competence_levels = bool(show_competence_levels)

    def _persist_active_dataset_metadata(self) -> None:
        payload = {
            "dataset_id": self.active_dataset_id,
            "dataset_path": self.active_dataset_path,
            "dataset_name": self.active_dataset_name,
            "target_column": self.target_column,
            "selected_feature_columns": self.selected_feature_columns,
            "feature_chart_settings": self.feature_chart_settings,
            "show_missing_bars": bool(self.show_missing_bars),
            "show_competence_levels": bool(self.show_competence_levels),
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

    def inspect_uploaded_dataset(
        self,
        file_name: str,
        file_bytes: bytes,
        *,
        use_case_study_feature_names: bool = False,
    ) -> Dict[str, Any]:
        """Persist an uploaded CSV and return column preview information."""
        if not file_name or not file_name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported")
        if not file_bytes:
            raise ValueError("Uploaded file is empty")

        dataset_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex}.csv"
        dataset_path = self.datasets_dir / dataset_id
        dataset_path.write_bytes(file_bytes)

        try:
            if use_case_study_feature_names:
                case_study_df = pd.read_csv(dataset_path)
                renamed_columns = {
                    source_name: exported_name
                    for source_name, exported_name in (
                        CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()
                    )
                    if source_name in case_study_df.columns
                }
                if renamed_columns:
                    case_study_df = case_study_df.rename(columns=renamed_columns)
                    case_study_df.to_csv(dataset_path, index=False)
            preview_df = pd.read_csv(dataset_path, nrows=MAX_PREVIEW_ROWS)
        except Exception as exc:
            dataset_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to parse CSV: {exc}") from exc

        columns = [str(col) for col in preview_df.columns]
        if len(columns) < 2:
            dataset_path.unlink(missing_ok=True)
            raise ValueError("CSV must contain at least one target and one feature column")

        default_target = (
            CASE_STUDY_TARGET_COLUMN
            if use_case_study_feature_names and CASE_STUDY_TARGET_COLUMN in columns
            else ("cnt" if "cnt" in columns else columns[-1])
        )
        return {
            "dataset_id": dataset_id,
            "original_filename": file_name,
            "columns": columns,
            "default_target_column": default_target,
            "case_study_feature_columns": (
                [name for name in CASE_STUDY_FEATURE_NAMES if name in columns]
                if use_case_study_feature_names
                else None
            ),
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

    @staticmethod
    def _normalize_scale_variable_name(variable_name: Any) -> str:
        normalized = str(variable_name or "").strip()
        if not normalized:
            return ""
        return re.sub(r"_\d+$", "", normalized)

    @staticmethod
    def _normalize_scale_topic(scale_label: Any) -> str:
        normalized = str(scale_label or "").strip()
        if not normalized:
            return ""
        normalized = re.sub(r"^(skala|scale)\s*-\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s*\(imputiert\)\s*$", "", normalized, flags=re.IGNORECASE)
        return normalized.strip()

    @staticmethod
    def _shared_response_options_from_details(
        detail_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        common_options: Optional[List[Dict[str, Any]]] = None

        for detail in detail_rows:
            options = _json_safe(detail.get("response_options", []))
            if not options:
                continue

            if common_options is None:
                common_options = options
            elif common_options != options:
                return []

        return common_options or []

    @staticmethod
    def _scale_topic_tokens(scale_topic: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", str(scale_topic or "").lower())
            if token
        ]

    def _infer_iqb_scale_item_details(
        self,
        source_variables: List[str],
        source_labels: List[str],
        response_option_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        inferred_details: List[Dict[str, Any]] = []
        seen_variables: set[str] = set()

        def append_candidate(variable: str, metadata: Dict[str, Any]) -> None:
            if variable in seen_variables:
                return

            inferred_details.append(
                {
                    "variable": variable,
                    "label": str(metadata.get("description", "") or "").strip(),
                    "response_options": _json_safe(
                        metadata.get("response_options", [])
                    ),
                    "missing_response_options": _json_safe(
                        metadata.get("missing_response_options", [])
                    ),
                }
            )
            seen_variables.add(variable)

        for index, source_variable in enumerate(source_variables):
            base_variable = self._normalize_scale_variable_name(source_variable)
            if not base_variable:
                continue

            scale_topic = self._normalize_scale_topic(
                source_labels[index] if index < len(source_labels) else "",
            )
            scale_topic_lower = scale_topic.lower()
            scale_topic_tokens = self._scale_topic_tokens(scale_topic)

            for candidate_variable, metadata in response_option_map.items():
                if candidate_variable in seen_variables:
                    continue
                if candidate_variable == source_variable or candidate_variable == base_variable:
                    continue
                if re.fullmatch(rf"{re.escape(base_variable)}_\d+", candidate_variable):
                    continue
                if not candidate_variable.startswith(base_variable):
                    continue

                response_options = _json_safe(metadata.get("response_options", []))
                if not response_options:
                    continue

                description = str(metadata.get("description", "") or "").strip()
                if not description:
                    continue
                if description.lower().startswith(("skala -", "scale -")):
                    continue
                if scale_topic_lower and scale_topic_lower not in description.lower():
                    continue

                append_candidate(candidate_variable, metadata)

            if inferred_details or not scale_topic_tokens:
                continue

            for candidate_variable, metadata in response_option_map.items():
                if candidate_variable in seen_variables:
                    continue
                if candidate_variable == source_variable or candidate_variable == base_variable:
                    continue

                response_options = _json_safe(metadata.get("response_options", []))
                if not response_options:
                    continue

                description = str(metadata.get("description", "") or "").strip()
                description_lower = description.lower()
                if not description:
                    continue
                if description_lower.startswith(("skala -", "scale -")):
                    continue
                if not all(token in description_lower for token in scale_topic_tokens):
                    continue

                append_candidate(candidate_variable, metadata)

        inferred_details.sort(key=lambda detail: str(detail.get("variable", "")))
        return inferred_details

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
            construction_type = self._classify_feature_construction(
                transformation,
                source_labels,
            )

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

            scale_item_details: List[Dict[str, Any]] = []
            if construction_type == "iqb_scale":
                scale_item_details = self._infer_iqb_scale_item_details(
                    source_variables,
                    source_labels,
                    response_option_map,
                )
                if not common_response_options:
                    common_response_options = self._shared_response_options_from_details(
                        scale_item_details,
                    )

            provenance_map[output_column] = {
                "role": str(row.get("role", "") or "").strip(),
                "category": str(row.get("category", "") or "").strip(),
                "source_variables": source_variables,
                "source_labels": source_labels,
                "source_details": source_details,
                "scale_item_details": scale_item_details,
                "transformation": transformation,
                "selection_rationale": selection_rationale,
                "construction_type": construction_type,
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
        display_title = str(raw_setting.get("display_title", "") or "").strip()
        x_axis_label = str(raw_setting.get("x_axis_label", "") or "").strip()
        categorical_value_labels = cls._sanitize_chart_label_map(
            raw_setting.get("categorical_value_labels", raw_setting.get("value_labels")),
        )
        numeric_tick_labels = cls._sanitize_chart_label_map(
            raw_setting.get("numeric_tick_labels"),
        )
        return {
            "display_title": display_title,
            "x_axis_label": x_axis_label,
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
                "display_title": "",
                "x_axis_label": "",
                "treat_as_categorical": False,
                "treat_as_numeric": False,
                "categorical_value_labels": {},
                "numeric_tick_labels": {},
            }
        return normalized

    def _get_feature_display_title(self, feature_name: str) -> str:
        stored = self._get_feature_chart_setting(feature_name)
        display_title = str(stored.get("display_title", "") or "").strip()
        return display_title or feature_name

    def _with_feature_display_name(
        self,
        feature_schema_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = _json_safe(feature_schema_entry)
        feature_name = str(enriched.get("name", "") or "").strip()
        if feature_name:
            enriched["display_name"] = self._get_feature_display_title(feature_name)
        return enriched

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
            "show_competence_levels": bool(self.show_competence_levels),
        }

    def update_chart_display_settings(
        self,
        show_missing_bars: bool,
        show_competence_levels: bool,
    ) -> Dict[str, Any]:
        self.show_missing_bars = bool(show_missing_bars)
        self.show_competence_levels = bool(show_competence_levels)
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
            "display_title": str(stored.get("display_title", "") or "").strip(),
            "x_axis_label": str(stored.get("x_axis_label", "") or "").strip(),
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
        display_title: Optional[str] = None,
        x_axis_label: Optional[str] = None,
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

        sanitized_display_title = (
            str(current.get("display_title", "") or "").strip()
            if display_title is None
            else str(display_title or "").strip()
        )
        sanitized_x_axis_label = (
            str(current.get("x_axis_label", "") or "").strip()
            if x_axis_label is None
            else str(x_axis_label or "").strip()
        )

        normalized_setting = {
            "display_title": sanitized_display_title,
            "x_axis_label": sanitized_x_axis_label,
            "treat_as_categorical": normalized_treat_as_categorical,
            "treat_as_numeric": normalized_treat_as_numeric,
            "categorical_value_labels": sanitized_categorical_labels,
            "numeric_tick_labels": sanitized_numeric_labels,
        }

        if (
            not normalized_setting["display_title"]
            and not normalized_setting["x_axis_label"]
            and not normalized_setting["treat_as_categorical"]
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

    def _get_legacy_compatible_export_shape_values(
        self,
        feature_name: str,
        x_values: List[Any],
        y_values: List[Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Use historic export grids for numeric 1--4 questionnaire scales.

        Older artifacts used a measurement-specific grid for each known scale.
        Keeping those grids makes new exports directly comparable while leaving
        other numeric domains (for example age or ISEI) untouched. Unknown 1--4
        scales use the historic 16-point default grid.
        """
        schema = self.feature_schema_map.get(feature_name, {}) or {}
        try:
            schema_min = float(schema.get("min_value"))
            schema_max = float(schema.get("max_value"))
        except (TypeError, ValueError):
            return x_values, y_values

        if not (
            np.isclose(schema_min, 1.0)
            and np.isclose(schema_max, 4.0)
            and len(x_values) == len(y_values)
        ):
            return x_values, y_values

        try:
            source_x = np.asarray(x_values, dtype=float)
            source_y = np.asarray(y_values, dtype=float)
        except (TypeError, ValueError):
            return x_values, y_values

        valid = np.isfinite(source_x) & np.isfinite(source_y)
        source_x = source_x[valid]
        source_y = source_y[valid]
        if source_x.size < 2:
            return x_values, y_values

        sort_order = np.argsort(source_x)
        source_x = source_x[sort_order]
        source_y = source_y[sort_order]
        unique_x, unique_indices = np.unique(source_x, return_index=True)
        source_y = source_y[unique_indices]
        endpoint_tolerance = 1e-6
        if (
            unique_x.size < 2
            or unique_x[0] > 1.0 + endpoint_tolerance
            or unique_x[-1] < 4.0 - endpoint_tolerance
        ):
            return x_values, y_values

        legacy_grid = np.asarray(
            LEGACY_NUMERIC_EXPORT_GRIDS.get(
                feature_name,
                DEFAULT_LEGACY_ONE_TO_FOUR_EXPORT_GRID,
            ),
            dtype=float,
        )
        return legacy_grid.tolist(), np.interp(legacy_grid, unique_x, source_y).tolist()

    def _export_shape_functions_payload(self) -> List[Dict[str, Any]]:
        if self.model_source == "imported":
            if isinstance(self.model, ImportedIGANNRuntime):
                return self.model.export_shape_functions()
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
            if feature_type == "numeric":
                (
                    exported_shape["x_values"],
                    exported_shape["y_values"],
                ) = self._get_legacy_compatible_export_shape_values(
                    feature_name,
                    exported_shape["x_values"],
                    exported_shape["y_values"],
                )
            if feature_type == "categorical":
                exported_shape["x_tick_labels"] = [
                    str(value) for value in _json_safe(raw_shape.get("x", []))
                ]
            distribution = self._build_shape_function_distribution(
                feature_name,
                feature_type,
                exported_shape["x_values"],
                x_tick_labels=exported_shape.get("x_tick_labels"),
            )
            if distribution is not None:
                exported_shape["distribution"] = _json_safe(distribution)
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

    def _export_held_out_test_data(self) -> Optional[Dict[str, Any]]:
        """Serialize the exact preprocessed test split used during training."""
        if (
            self.X_test is None
            or self.y_test is None
            or not self.feature_names
            or self.X_test.empty
            or self.y_test.empty
        ):
            return None

        feature_rows = self.X_test.loc[:, self.feature_names].to_dict(
            orient="records"
        )
        target_values = self.y_test.iloc[:, 0].tolist()
        if len(feature_rows) != len(target_values):
            return None

        return {
            "feature_rows": _json_safe(feature_rows),
            "target_values": _json_safe(target_values),
            "sample_ids": [str(index) for index in self.X_test.index],
        }

    def _export_training_feature_rows(self) -> Optional[List[Dict[str, Any]]]:
        """Serialize preprocessed training rows for artifact-based mean predictions."""
        if (
            self.X_train is not None
            and not self.X_train.empty
            and self.feature_names
            and set(self.feature_names).issubset(self.X_train.columns)
        ):
            return _json_safe(
                self.X_train.loc[:, self.feature_names].to_dict(orient="records")
            )
        if self.model_source == "imported":
            return self.imported_training_feature_rows
        return None

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

        shape_functions = [
            {
                **shape_function,
                "chart_config": {
                    key: value
                    for key, value in ((shape_function.get("chart_config") or {}).items())
                    if key not in {"display_title", "x_axis_label"}
                },
            }
            for shape_function in self._export_shape_functions_payload()
        ]
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
                MODEL_ARTIFACT_VERSION_WITH_NOOP_EDITS
                if include_shape_function_edits
                else MODEL_ARTIFACT_VERSION
            ),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "IGANN",
            "model_source": self.model_source,
            "igann_params": igann_params,
            "preprocessing_scope": "whole_dataset_before_split" if self.model_source == "trained" else "unknown",
            "effective_boosting_rounds": int(len(getattr(self.model, "boosting_rates", []) or [])),
            "base_prediction_offset": float(self._infer_base_prediction_offset()),
            "feature_names": exported_feature_names,
            "selected_feature_columns": list(self.selected_feature_columns),
            "target_column": self.target_column,
            "feature_schema": exported_feature_schema,
            "cat_features": exported_cat_features,
            "num_features": exported_num_features,
            "missing_indicator_map": _json_safe(self.missing_indicator_map),
            "numeric_missing_placeholder_values": _json_safe(
                self.numeric_missing_placeholder_values
            ),
            "feature_chart_settings": _json_safe(
                {
                    feature_name: {
                        key: value
                        for key, value in (setting or {}).items()
                        if key not in {"display_title", "x_axis_label"}
                    }
                    for feature_name, setting in self.feature_chart_settings.items()
                }
            ),
            "show_missing_bars": bool(self.show_missing_bars),
            "show_competence_levels": bool(self.show_competence_levels),
            "shape_functions": shape_functions,
            "dataset_id": self.active_dataset_id,
            "dataset_name": self.active_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None),
        }
        held_out_test_data = self._export_held_out_test_data()
        if held_out_test_data is not None:
            artifact["held_out_test_data"] = held_out_test_data
        training_feature_rows = self._export_training_feature_rows()
        if training_feature_rows is not None:
            artifact["training_feature_rows"] = training_feature_rows
            if self.X_train is not None and len(training_feature_rows) == len(self.X_train):
                artifact["training_sample_ids"] = [str(index) for index in self.X_train.index]
        if held_out_test_data is not None and (self.active_dataset_id or self.active_dataset_name):
            artifact["sample_id_namespace"] = str(
                self.active_dataset_id or self.active_dataset_name
            )
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
        distribution = shape_function.get("distribution")
        if distribution is not None:
            if not isinstance(distribution, dict):
                raise ValueError(
                    f"Shape function '{feature_name}' distribution must be an object"
                )
            chart_type = str(distribution.get("chart_type", "")).strip()
            if chart_type not in {"numeric", "categorical"}:
                raise ValueError(
                    f"Shape function '{feature_name}' distribution has an invalid chart_type"
                )
            bins = distribution.get("bins", [])
            counts = distribution.get("counts", [])
            if not isinstance(bins, list) or not isinstance(counts, list):
                raise ValueError(
                    f"Shape function '{feature_name}' distribution bins and counts must be arrays"
                )
            normalized["distribution"] = _json_safe(distribution)
        x_tick_labels = shape_function.get("x_tick_labels")
        if isinstance(x_tick_labels, list):
            normalized["x_tick_labels"] = [str(label) for label in x_tick_labels]
        return normalized

    def _normalize_imported_public_shape_function(
        self,
        shape_function: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_shape = _json_safe(shape_function)

        feature_name = str(normalized_shape.get("feature_name", "")).strip()
        if not feature_name:
            return normalized_shape

        stored_chart_config = _json_safe(self.feature_chart_settings.get(feature_name) or {})
        chart_config = _json_safe(normalized_shape.get("chart_config") or {})
        if not str(chart_config.get("display_title", "") or "").strip():
            chart_config["display_title"] = stored_chart_config.get("display_title", "")
        if not str(chart_config.get("x_axis_label", "") or "").strip():
            chart_config["x_axis_label"] = stored_chart_config.get("x_axis_label", "")
        if not (chart_config.get("categorical_value_labels") or {}):
            chart_config["categorical_value_labels"] = stored_chart_config.get(
                "categorical_value_labels",
                {},
            )
        if not (chart_config.get("numeric_tick_labels") or {}):
            chart_config["numeric_tick_labels"] = stored_chart_config.get(
                "numeric_tick_labels",
                {},
            )
        if "treat_as_categorical" not in chart_config:
            chart_config["treat_as_categorical"] = stored_chart_config.get(
                "treat_as_categorical",
                False,
            )
        if "treat_as_numeric" not in chart_config:
            chart_config["treat_as_numeric"] = stored_chart_config.get(
                "treat_as_numeric",
                False,
            )
        normalized_shape["chart_config"] = chart_config

        if (
            str(normalized_shape.get("feature_type")) != "numeric"
            or str(chart_config.get("chart_feature_type", "numeric")) != "numeric"
        ):
            return normalized_shape

        x_values = normalized_shape.get("x_values") or []
        y_values = normalized_shape.get("y_values") or []
        if (
            not isinstance(x_values, list)
            or not isinstance(y_values, list)
            or len(x_values) != len(y_values)
            or not x_values
        ):
            return normalized_shape

        try:
            x_numeric = np.asarray(x_values, dtype=float)
            y_numeric = np.asarray(y_values, dtype=float)
        except (TypeError, ValueError):
            return normalized_shape

        if x_numeric.size == 0 or y_numeric.size == 0 or x_numeric.size != y_numeric.size:
            return normalized_shape

        order = np.argsort(x_numeric)
        x_numeric = x_numeric[order]
        y_numeric = y_numeric[order]

        schema = self.feature_schema_map.get(feature_name, {}) or {}
        schema_min = schema.get("min_value")
        schema_max = schema.get("max_value")
        try:
            domain_min = float(schema_min)
            domain_max = float(schema_max)
        except (TypeError, ValueError):
            return normalized_shape

        if (
            not np.isfinite(domain_min)
            or not np.isfinite(domain_max)
            or domain_max < domain_min
        ):
            return normalized_shape

        if abs(domain_max - domain_min) < 1e-12:
            target_x = np.asarray([domain_min], dtype=float)
        else:
            target_x = np.linspace(domain_min, domain_max, 30)

        normalized_shape["x_values"] = [float(value) for value in target_x.tolist()]
        normalized_shape["y_values"] = [
            float(value) for value in np.interp(target_x, x_numeric, y_numeric).tolist()
        ]

        numeric_tick_labels = chart_config.get("numeric_tick_labels") or {}
        x_tick_labels = self._build_numeric_chart_tick_labels(
            normalized_shape["x_values"],
            numeric_tick_labels,
        )
        if x_tick_labels is not None:
            normalized_shape["x_tick_labels"] = x_tick_labels
        else:
            normalized_shape.pop("x_tick_labels", None)

        return normalized_shape

    @staticmethod
    def _infer_missing_indicator_map_from_artifact(
        feature_names: List[str],
        selected_feature_columns: List[str],
    ) -> Dict[str, str]:
        selected_feature_set = {
            str(feature_name).strip()
            for feature_name in selected_feature_columns
            if str(feature_name).strip()
        }
        inferred: Dict[str, str] = {}

        for feature_name in feature_names:
            normalized_name = str(feature_name).strip()
            if not normalized_name.endswith(MISSING_INDICATOR_SUFFIX):
                continue
            base_feature_name = normalized_name[: -len(MISSING_INDICATOR_SUFFIX)].strip()
            if base_feature_name and base_feature_name in selected_feature_set:
                inferred[base_feature_name] = normalized_name

        return inferred

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
                    "is_imputed_noop": bool(shape_function.get("is_imputed_noop", False)),
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

    @staticmethod
    def _validate_held_out_test_data(
        raw_test_data: Any,
        *,
        feature_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Validate optional test rows embedded in an exported model artifact."""
        if raw_test_data is None:
            return None
        if not isinstance(raw_test_data, dict):
            raise ValueError("Imported artifact held_out_test_data must be an object")

        feature_rows = raw_test_data.get("feature_rows")
        target_values = raw_test_data.get("target_values")
        if not isinstance(feature_rows, list) or not isinstance(target_values, list):
            raise ValueError(
                "Imported artifact held_out_test_data must include feature_rows and target_values arrays"
            )
        if not feature_rows or len(feature_rows) != len(target_values):
            raise ValueError(
                "Imported artifact held_out_test_data must contain equally sized non-empty rows and targets"
            )

        expected_features = set(feature_names)
        normalized_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(feature_rows):
            if not isinstance(row, dict) or set(row) != expected_features:
                raise ValueError(
                    "Imported artifact held_out_test_data row "
                    f"{index} must contain exactly the model feature_names"
                )
            normalized_rows.append(_json_safe(row))

        normalized_targets: List[float] = []
        for index, value in enumerate(target_values):
            try:
                target = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Imported artifact held_out_test_data target_values must be numeric "
                    f"(invalid value at index {index})"
                ) from exc
            if not np.isfinite(target):
                raise ValueError(
                    "Imported artifact held_out_test_data target_values must be finite "
                    f"(invalid value at index {index})"
                )
            normalized_targets.append(target)

        sample_ids = raw_test_data.get("sample_ids")
        if sample_ids is not None:
            if (
                not isinstance(sample_ids, list)
                or len(sample_ids) != len(feature_rows)
                or any(not str(value).strip() for value in sample_ids)
                or len({str(value) for value in sample_ids}) != len(sample_ids)
            ):
                raise ValueError("Imported artifact held_out_test_data sample_ids must be unique and match the rows")

        result = {
            "feature_rows": normalized_rows,
            "target_values": normalized_targets,
        }
        if sample_ids is not None:
            result["sample_ids"] = [str(value) for value in sample_ids]
        return result

    @staticmethod
    def _validate_training_feature_rows(
        raw_rows: Any,
        *,
        feature_names: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        if raw_rows is None:
            return None
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("Imported artifact training_feature_rows must be a non-empty array")
        expected_features = set(feature_names)
        normalized_rows = []
        for index, row in enumerate(raw_rows):
            if not isinstance(row, dict) or set(row) != expected_features:
                raise ValueError(
                    "Imported artifact training_feature_rows row "
                    f"{index} must contain exactly the model feature_names"
                )
            normalized_rows.append(_json_safe(row))
        return normalized_rows

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
        missing_indicator_map = artifact.get("missing_indicator_map")
        numeric_missing_placeholder_values = artifact.get(
            "numeric_missing_placeholder_values"
        )

        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("Imported artifact must include non-empty feature_names")
        if not isinstance(selected_feature_columns, list) or not selected_feature_columns:
            raise ValueError("Imported artifact must include non-empty selected_feature_columns")
        if not isinstance(cat_features, list) or not isinstance(num_features, list):
            raise ValueError("Imported artifact must include cat_features and num_features arrays")
        normalized_feature_names = [str(item) for item in feature_names]
        selected_feature_names = [str(item) for item in selected_feature_columns]
        if sorted([str(item) for item in cat_features] + [str(item) for item in num_features]) != sorted(normalized_feature_names):
            raise ValueError("Imported artifact cat_features and num_features must cover all feature_names exactly")
        if not isinstance(shape_functions, list) or len(shape_functions) != len(feature_names):
            raise ValueError("Imported artifact must include one shape function per feature")
        if not set(selected_feature_names).issubset(set(normalized_feature_names)):
            raise ValueError(
                "Imported artifact selected_feature_columns must be covered by feature_names"
            )
        if not isinstance(feature_schema, list) or not feature_schema:
            raise ValueError("Imported artifact must include non-empty feature_schema")

        feature_schema_map = {}
        for item in feature_schema:
            if not isinstance(item, dict):
                raise ValueError("Imported artifact feature_schema entries must be objects")
            feature_name = str(item.get("name", "")).strip()
            feature_type = str(item.get("feature_type", "")).strip()
            if feature_name not in selected_feature_names:
                if feature_name in normalized_feature_names:
                    continue
                raise ValueError(
                    f"Imported artifact schema references unknown feature '{feature_name}'"
                )
            if feature_type not in {"numeric", "categorical"}:
                raise ValueError(f"Imported artifact schema has invalid feature_type for '{feature_name}'")
            feature_schema_map[feature_name] = _json_safe(item)

        if sorted(feature_schema_map.keys()) != sorted(selected_feature_names):
            raise ValueError(
                "Imported artifact feature_schema must cover all selected_feature_columns exactly"
            )

        if missing_indicator_map is None:
            normalized_missing_indicator_map = (
                self._infer_missing_indicator_map_from_artifact(
                    normalized_feature_names,
                    selected_feature_names,
                )
            )
        else:
            if not isinstance(missing_indicator_map, dict):
                raise ValueError(
                    "Imported artifact missing_indicator_map must be an object when present"
                )
            normalized_missing_indicator_map = {}
            for raw_feature_name, raw_indicator_name in missing_indicator_map.items():
                feature_name = str(raw_feature_name).strip()
                indicator_name = str(raw_indicator_name).strip()
                if feature_name not in selected_feature_names:
                    raise ValueError(
                        f"Imported artifact missing_indicator_map references unknown feature '{feature_name}'"
                    )
                if indicator_name not in normalized_feature_names:
                    raise ValueError(
                        f"Imported artifact missing_indicator_map references unknown indicator '{indicator_name}'"
                    )
                normalized_missing_indicator_map[feature_name] = indicator_name

        for feature_name, indicator_name in normalized_missing_indicator_map.items():
            if indicator_name in selected_feature_names:
                raise ValueError(
                    f"Imported artifact missing indicator '{indicator_name}' must not be a public feature"
                )
            if str(feature_schema_map.get(feature_name, {}).get("feature_type")) != "numeric":
                raise ValueError(
                    f"Imported artifact missing indicator '{indicator_name}' must map to a numeric feature"
                )

        allowed_internal_features = set(selected_feature_names) | set(
            normalized_missing_indicator_map.values()
        )
        unexpected_internal_features = sorted(
            feature_name
            for feature_name in normalized_feature_names
            if feature_name not in allowed_internal_features
        )
        if unexpected_internal_features:
            raise ValueError(
                "Imported artifact feature_names contains unsupported internal features: "
                f"{unexpected_internal_features}"
            )

        if not set(normalized_missing_indicator_map.values()).issubset(
            set(str(item) for item in cat_features)
        ):
            raise ValueError(
                "Imported artifact missing indicators must be listed in cat_features"
            )

        normalized_shape_functions = [
            self._validate_shape_function_artifact(item)
            for item in shape_functions
        ]
        if sorted(sf["feature_name"] for sf in normalized_shape_functions) != sorted(normalized_feature_names):
            raise ValueError("Imported artifact shape_functions must cover all feature_names exactly")

        public_shape_functions = [
            shape_function
            for shape_function in normalized_shape_functions
            if shape_function["feature_name"] in selected_feature_names
        ]
        if sorted(sf["feature_name"] for sf in public_shape_functions) != sorted(
            selected_feature_names
        ):
            raise ValueError(
                "Imported artifact shape_functions must cover all selected_feature_columns exactly"
            )

        if numeric_missing_placeholder_values is not None and not isinstance(
            numeric_missing_placeholder_values,
            dict,
        ):
            raise ValueError(
                "Imported artifact numeric_missing_placeholder_values must be an object when present"
            )
        normalized_numeric_missing_placeholder_values: Dict[str, float] = {}
        for feature_name in normalized_missing_indicator_map:
            fallback_value = feature_schema_map.get(feature_name, {}).get(
                "default_value",
                0.0,
            )
            raw_value = (
                numeric_missing_placeholder_values.get(feature_name, fallback_value)
                if isinstance(numeric_missing_placeholder_values, dict)
                else fallback_value
            )
            try:
                normalized_numeric_missing_placeholder_values[feature_name] = float(
                    raw_value
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Imported artifact numeric_missing_placeholder_values must be numeric "
                    f"for '{feature_name}'"
                ) from exc

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
            feature_names=normalized_feature_names,
        ) if shape_function_edits_export is not None else {}
        held_out_test_data = self._validate_held_out_test_data(
            artifact.get("held_out_test_data"),
            feature_names=normalized_feature_names,
        )
        training_feature_rows = self._validate_training_feature_rows(
            artifact.get("training_feature_rows"),
            feature_names=normalized_feature_names,
        )
        # Older exports round the schema default (and sometimes the explicit
        # placeholder metadata) to two decimals. The exported, preprocessed
        # training rows retain the exact fitted fill used by the model.
        for feature_name, indicator_name in normalized_missing_indicator_map.items():
            missing_rows = [
                row for row in (training_feature_rows or [])
                if row.get(indicator_name) == MISSING_CATEGORY_VALUE
            ]
            if not missing_rows:
                continue
            try:
                fills = [float(row[feature_name]) for row in missing_rows]
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(value) and value == fills[0] for value in fills):
                continue
            declared = normalized_numeric_missing_placeholder_values[feature_name]
            if (
                np.isclose(declared, fills[0], rtol=0, atol=1e-8)
                or np.isclose(declared, round(fills[0], 2), rtol=0, atol=1e-8)
            ):
                normalized_numeric_missing_placeholder_values[feature_name] = fills[0]
        training_sample_ids = artifact.get("training_sample_ids")
        if training_sample_ids is not None:
            if (
                not isinstance(training_sample_ids, list)
                or any(not str(value).strip() for value in training_sample_ids)
                or len({str(value) for value in training_sample_ids}) != len(training_sample_ids)
                or (training_feature_rows is not None and len(training_sample_ids) != len(training_feature_rows))
            ):
                raise ValueError("Imported artifact training_sample_ids must be unique and match training rows")
            training_sample_ids = [str(value) for value in training_sample_ids]
        sample_id_namespace = artifact.get("sample_id_namespace")
        if sample_id_namespace is not None:
            sample_id_namespace = str(sample_id_namespace).strip()
            if not sample_id_namespace:
                raise ValueError("Imported artifact sample_id_namespace must be non-empty")
        preprocessing_scope = artifact.get("preprocessing_scope")
        if preprocessing_scope is not None and preprocessing_scope not in {
            "whole_dataset_before_split", "training_partition_only", "unknown"
        }:
            raise ValueError("Imported artifact preprocessing_scope is invalid")

        return {
            "artifact_version": artifact_version,
            "igann_params": dict(artifact.get("igann_params") or {}),
            "effective_boosting_rounds": int(artifact.get("effective_boosting_rounds") or 0),
            "base_prediction_offset": float(artifact.get("base_prediction_offset")),
            "feature_names": normalized_feature_names,
            "selected_feature_columns": selected_feature_names,
            "target_column": str(artifact.get("target_column")),
            "feature_schema": list(feature_schema_map.values()),
            "feature_schema_map": feature_schema_map,
            "cat_features": [str(item) for item in cat_features],
            "num_features": [str(item) for item in num_features],
            "missing_indicator_map": normalized_missing_indicator_map,
            "numeric_missing_placeholder_values": (
                normalized_numeric_missing_placeholder_values
            ),
            "feature_chart_settings": {
                str(feature_name): normalized
                for feature_name, raw_setting in (
                    _json_safe(artifact.get("feature_chart_settings") or {})
                ).items()
                for normalized in [self._normalize_stored_feature_chart_setting(raw_setting)]
                if normalized
            },
            "show_missing_bars": bool(artifact.get("show_missing_bars", False)),
            "show_competence_levels": bool(
                artifact.get("show_competence_levels", False)
            ),
            "shape_functions": normalized_shape_functions,
            "public_shape_functions": public_shape_functions,
            "dataset_id": artifact.get("dataset_id"),
            "dataset_name": artifact.get("dataset_name"),
            "shape_function_edits_export": normalized_shape_function_edits_export,
            "held_out_test_data": held_out_test_data,
            "training_feature_rows": training_feature_rows,
            "training_sample_ids": training_sample_ids,
            "sample_id_namespace": sample_id_namespace,
            "preprocessing_scope": preprocessing_scope,
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
        self.imported_training_feature_rows = validated["training_feature_rows"]
        self.df = None
        self.is_trained = True
        self.feature_names = list(validated["feature_names"])
        self.selected_feature_columns = list(validated["selected_feature_columns"])
        self.cat_features = list(validated["cat_features"])
        self.num_features = list(validated["num_features"])
        self.target_column = validated["target_column"]
        self.feature_schema = list(validated["feature_schema"])
        self.feature_schema_map = dict(validated["feature_schema_map"])
        self.missing_indicator_map = dict(validated["missing_indicator_map"])
        self.numeric_missing_placeholder_values = dict(
            validated["numeric_missing_placeholder_values"]
        )
        self.feature_chart_settings = dict(validated["feature_chart_settings"])
        self.show_missing_bars = bool(validated.get("show_missing_bars", False))
        self.show_competence_levels = bool(
            validated.get("show_competence_levels", False)
        )
        self.original_shape_functions = {
            shape_function["feature_name"]: self._normalize_imported_public_shape_function(
                shape_function
            )
            for shape_function in validated["public_shape_functions"]
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

    def _normalize_imported_public_shape_function_for_context(
        self,
        shape_function: Dict[str, Any],
        *,
        feature_chart_settings: Dict[str, Dict[str, Any]],
        feature_schema_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized_shape = _json_safe(shape_function)

        feature_name = str(normalized_shape.get("feature_name", "")).strip()
        if not feature_name:
            return normalized_shape

        stored_chart_config = _json_safe(
            (feature_chart_settings or {}).get(feature_name) or {}
        )
        chart_config = _json_safe(normalized_shape.get("chart_config") or {})
        if not str(chart_config.get("display_title", "") or "").strip():
            chart_config["display_title"] = stored_chart_config.get("display_title", "")
        if not str(chart_config.get("x_axis_label", "") or "").strip():
            chart_config["x_axis_label"] = stored_chart_config.get("x_axis_label", "")
        if not (chart_config.get("categorical_value_labels") or {}):
            chart_config["categorical_value_labels"] = stored_chart_config.get(
                "categorical_value_labels",
                {},
            )
        if not (chart_config.get("numeric_tick_labels") or {}):
            chart_config["numeric_tick_labels"] = stored_chart_config.get(
                "numeric_tick_labels",
                {},
            )
        if "treat_as_categorical" not in chart_config:
            chart_config["treat_as_categorical"] = stored_chart_config.get(
                "treat_as_categorical",
                False,
            )
        if "treat_as_numeric" not in chart_config:
            chart_config["treat_as_numeric"] = stored_chart_config.get(
                "treat_as_numeric",
                False,
            )
        normalized_shape["chart_config"] = chart_config

        if (
            str(normalized_shape.get("feature_type")) != "numeric"
            or str(chart_config.get("chart_feature_type", "numeric")) != "numeric"
        ):
            return normalized_shape

        x_values = normalized_shape.get("x_values") or []
        y_values = normalized_shape.get("y_values") or []
        if (
            not isinstance(x_values, list)
            or not isinstance(y_values, list)
            or len(x_values) != len(y_values)
            or not x_values
        ):
            return normalized_shape

        try:
            x_numeric = np.asarray(x_values, dtype=float)
            y_numeric = np.asarray(y_values, dtype=float)
        except (TypeError, ValueError):
            return normalized_shape

        if x_numeric.size == 0 or y_numeric.size == 0 or x_numeric.size != y_numeric.size:
            return normalized_shape

        order = np.argsort(x_numeric)
        x_numeric = x_numeric[order]
        y_numeric = y_numeric[order]

        schema = (feature_schema_map or {}).get(feature_name, {}) or {}
        schema_min = schema.get("min_value")
        schema_max = schema.get("max_value")
        try:
            domain_min = float(schema_min)
            domain_max = float(schema_max)
        except (TypeError, ValueError):
            return normalized_shape

        if (
            not np.isfinite(domain_min)
            or not np.isfinite(domain_max)
            or domain_max < domain_min
        ):
            return normalized_shape

        if abs(domain_max - domain_min) < 1e-12:
            target_x = np.asarray([domain_min], dtype=float)
        else:
            target_x = np.linspace(domain_min, domain_max, 30)

        normalized_shape["x_values"] = [float(value) for value in target_x.tolist()]
        normalized_shape["y_values"] = [
            float(value) for value in np.interp(target_x, x_numeric, y_numeric).tolist()
        ]

        numeric_tick_labels = chart_config.get("numeric_tick_labels") or {}
        x_tick_labels = self._build_numeric_chart_tick_labels(
            normalized_shape["x_values"],
            numeric_tick_labels,
        )
        if x_tick_labels is not None:
            normalized_shape["x_tick_labels"] = x_tick_labels
        else:
            normalized_shape.pop("x_tick_labels", None)

        return normalized_shape

    @staticmethod
    def _format_compare_numeric_display(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _decode_compare_point_display_value(
        cls,
        feature_type: str,
        raw_value: Any,
        original_x: List[Any],
    ) -> Any:
        if feature_type == "categorical":
            return str(raw_value)
        decoded = cls._decode_numeric_storage_key(raw_value, original_x)
        if decoded is not None:
            return float(decoded)
        return raw_value

    @classmethod
    def _compare_point_sort_key(
        cls,
        feature_type: str,
        raw_value: Any,
        original_x: List[Any],
    ) -> Any:
        if feature_type == "categorical":
            return (0, str(raw_value))
        decoded = cls._decode_numeric_storage_key(raw_value, original_x)
        if decoded is not None:
            return (0, float(decoded))
        try:
            return (0, float(raw_value))
        except (TypeError, ValueError):
            return (1, str(raw_value))

    @classmethod
    def _build_compare_x_summary(
        cls,
        feature_type: str,
        edited_points: List[Dict[str, Any]],
        original_x: List[Any],
    ) -> str:
        if not edited_points:
            return ""

        if feature_type == "categorical":
            labels = [str(point.get("display_x_value", point.get("x_value"))) for point in edited_points]
            unique_labels = list(dict.fromkeys(labels))
            if len(unique_labels) <= 3:
                return ", ".join(unique_labels)
            preview = ", ".join(unique_labels[:3])
            return f"{preview}, +{len(unique_labels) - 3} more"

        numeric_values: List[float] = []
        for point in edited_points:
            decoded = cls._decode_numeric_storage_key(point.get("x_value"), original_x)
            if decoded is None:
                continue
            numeric_values.append(float(decoded))

        numeric_values = sorted(numeric_values)
        if not numeric_values:
            return str(edited_points[0].get("display_x_value", edited_points[0].get("x_value", "")))
        if len(numeric_values) == 1:
            return cls._format_compare_numeric_display(numeric_values[0])
        if len(numeric_values) == 2:
            return ", ".join(
                cls._format_compare_numeric_display(value) for value in numeric_values
            )
        return (
            f"{cls._format_compare_numeric_display(numeric_values[0])}"
            f" to {cls._format_compare_numeric_display(numeric_values[-1])}"
        )

    @classmethod
    def _decode_numeric_offsets_for_shape_function(
        cls,
        shape_function: Dict[str, Any],
        offsets: Dict[Any, float],
    ) -> List[Tuple[float, float]]:
        original_x = shape_function.get("x_values", [])
        point_map: Dict[float, float] = {}
        for raw_key, raw_offset in (offsets or {}).items():
            x_val = cls._decode_numeric_storage_key(raw_key, original_x)
            if x_val is None:
                continue
            try:
                point_map[float(x_val)] = float(raw_offset)
            except (TypeError, ValueError):
                continue
        return sorted(point_map.items(), key=lambda item: item[0])

    @classmethod
    def _get_compare_offset_for_feature_value(
        cls,
        *,
        shape_function: Dict[str, Any],
        original_is_categorical: bool,
        value: Any,
        offsets: Optional[Dict[Any, float]] = None,
    ) -> float:
        offsets = offsets or {}
        if not offsets:
            return 0.0

        is_categorical_chart = str(shape_function.get("feature_type")) == "categorical"
        if is_categorical_chart:
            candidate_keys = [
                cls._stringify_chart_value(value),
                str(value),
            ]
            for raw_key in candidate_keys:
                if raw_key in offsets:
                    try:
                        return float(offsets.get(raw_key, 0.0))
                    except (TypeError, ValueError):
                        return 0.0

            points = cls._decode_numeric_offsets_for_shape_function(shape_function, offsets)
            if points:
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    return 0.0
                for x_val, x_offset in points:
                    if abs(x_val - value_float) < 1e-9:
                        return float(x_offset)
            return 0.0

        if original_is_categorical:
            candidate_keys = [
                cls._stringify_chart_value(value),
                str(value),
            ]
            for raw_key in candidate_keys:
                if raw_key in offsets:
                    try:
                        return float(offsets.get(raw_key, 0.0))
                    except (TypeError, ValueError):
                        return 0.0

        points = cls._decode_numeric_offsets_for_shape_function(shape_function, offsets)
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
                ratio = (value_float - left_x) / span
                return float(left_offset + ratio * (right_offset - left_offset))

        return float(offset_values[-1])

    def prepare_model_compare_artifact(
        self,
        artifact: Dict[str, Any],
        *,
        filename: str,
    ) -> Dict[str, Any]:
        validated = self._validate_import_artifact(artifact)
        feature_chart_settings = dict(validated["feature_chart_settings"])
        feature_schema_map = dict(validated["feature_schema_map"])

        raw_public_shape_map = {
            shape_function["feature_name"]: shape_function
            for shape_function in validated["public_shape_functions"]
        }
        prepared_shape_functions = []
        prepared_shape_map: Dict[str, Dict[str, Any]] = {}
        for feature_name in validated["selected_feature_columns"]:
            shape_function = raw_public_shape_map.get(feature_name)
            if not shape_function:
                continue
            normalized_shape = self._normalize_imported_public_shape_function_for_context(
                shape_function,
                feature_chart_settings=feature_chart_settings,
                feature_schema_map=feature_schema_map,
            )
            prepared_shape_functions.append(normalized_shape)
            prepared_shape_map[feature_name] = normalized_shape

        # Keep internal missing-indicator shape functions for held-out prediction
        # reconstruction while exposing only public shape functions in the UI.
        prepared_feature_names = {
            str(shape_function.get("feature_name"))
            for shape_function in prepared_shape_functions
        }
        prepared_shape_functions.extend(
            _json_safe(shape_function)
            for shape_function in validated["shape_functions"]
            if str(shape_function.get("feature_name")) not in prepared_feature_names
        )

        submissions_by_feature: Dict[str, Dict[str, Any]] = {
            feature_name: {
                "feature_name": feature_name,
                "feature_type": prepared_shape_map.get(feature_name, {}).get(
                    "feature_type",
                    "numeric",
                ),
                "submissions": [],
            }
            for feature_name in validated["selected_feature_columns"]
        }

        edits_export = validated.get("shape_function_edits_export", {}) or {}
        declared_professions = {
            str(user.get("name", "")).strip(): (
                str(user.get("profession", "")).strip() or None
            )
            for user in (edits_export.get("users") or [])
            if isinstance(user, dict) and str(user.get("name", "")).strip()
        }

        for user_index, user_group in enumerate(edits_export.get("edits") or []):
            user_name = str(user_group.get("user_name", "")).strip()
            if not user_name:
                continue

            for shape_index, shape_function in enumerate(
                user_group.get("shape_functions") or []
            ):
                feature_name = str(shape_function.get("feature_name", "")).strip()
                base_shape = prepared_shape_map.get(feature_name)
                if not feature_name or base_shape is None:
                    continue

                effective_feature_type = str(base_shape.get("feature_type", "numeric"))
                original_x = base_shape.get("x_values", [])
                edited_points = []
                for point in shape_function.get("edited_points") or []:
                    normalized_point = {
                        "x_value": point.get("x_value"),
                        "y_value": float(point.get("y_value", 0.0)),
                        "weight": float(point.get("weight", 0.5)),
                        "message": str(point.get("message", "") or ""),
                    }
                    normalized_point["display_x_value"] = self._decode_compare_point_display_value(
                        effective_feature_type,
                        normalized_point["x_value"],
                        original_x,
                    )
                    edited_points.append(normalized_point)

                if not edited_points:
                    continue

                edited_points.sort(
                    key=lambda point: self._compare_point_sort_key(
                        effective_feature_type,
                        point.get("x_value"),
                        original_x,
                    )
                )

                raw_submission_id = str(shape_function.get("submission_id", "") or "").strip()
                submission_id = (
                    raw_submission_id
                    or f"artifact-{user_index}-{shape_index}-{feature_name}"
                )
                avg_weight = sum(
                    float(point.get("weight", 0.5)) for point in edited_points
                ) / max(len(edited_points), 1)
                sureness = shape_function.get("sureness")
                try:
                    sureness_value = int(sureness)
                except (TypeError, ValueError):
                    sureness_value = round(avg_weight * 10)

                submissions_by_feature[feature_name]["submissions"].append(
                    {
                        "submission_id": submission_id,
                        "feature_name": feature_name,
                        "feature_type": effective_feature_type,
                        "user_name": user_name,
                        "profession": declared_professions.get(user_name),
                        "message": str(shape_function.get("message", "") or ""),
                        "is_imputed_noop": bool(shape_function.get("is_imputed_noop", False)),
                        "sureness": sureness_value,
                        "point_count": len(edited_points),
                        "x_summary": self._build_compare_x_summary(
                            effective_feature_type,
                            edited_points,
                            original_x,
                        ),
                        "created_at": shape_function.get("created_at"),
                        "updated_at": shape_function.get("updated_at"),
                        "edited_points": edited_points,
                    }
                )

        ordered_submissions_by_feature = []
        for feature_name in validated["selected_feature_columns"]:
            feature_group = submissions_by_feature.get(feature_name)
            if feature_group is None:
                continue
            feature_group["submissions"].sort(
                key=lambda submission: (
                    str(submission.get("created_at") or ""),
                    submission.get("submission_id", ""),
                ),
                reverse=True,
            )
            ordered_submissions_by_feature.append(feature_group)

        has_edit_export = bool(edits_export.get("included"))
        metadata = {
            "filename": filename,
            "exported_at": artifact.get("exported_at"),
            "dataset_name": validated.get("dataset_name"),
            "dataset_id": validated.get("dataset_id"),
            "target_column": validated["target_column"],
            "selected_feature_count": len(validated["selected_feature_columns"]),
            "has_edit_export": has_edit_export,
            "edit_user_count": int(edits_export.get("user_count", 0) or 0),
            "edit_submission_count": int(edits_export.get("submission_count", 0) or 0),
            "model_source": str(artifact.get("model_source", "") or ""),
            "artifact_version": validated["artifact_version"],
            "has_held_out_test_data": bool(validated.get("held_out_test_data")),
        }

        return {
            "metadata": metadata,
            "base_prediction_offset": validated["base_prediction_offset"],
            "feature_names": list(validated["feature_names"]),
            "selected_feature_columns": list(validated["selected_feature_columns"]),
            "cat_features": list(validated["cat_features"]),
            "num_features": list(validated["num_features"]),
            "missing_indicator_map": dict(validated["missing_indicator_map"]),
            "numeric_missing_placeholder_values": dict(
                validated["numeric_missing_placeholder_values"]
            ),
            "shape_functions": prepared_shape_functions,
            "submissions_by_feature": ordered_submissions_by_feature,
            "held_out_test_data": validated.get("held_out_test_data"),
            "training_feature_rows": validated.get("training_feature_rows"),
            "training_sample_ids": validated.get("training_sample_ids"),
            "sample_id_namespace": validated.get("sample_id_namespace"),
            "preprocessing_scope": validated.get("preprocessing_scope"),
        }

    @staticmethod
    def _validate_model_compare_pair(
        left_artifact: Dict[str, Any],
        right_artifact: Dict[str, Any],
    ) -> List[str]:
        left_target = str(
            ((left_artifact.get("metadata") or {}).get("target_column")) or ""
        )
        right_target = str(
            ((right_artifact.get("metadata") or {}).get("target_column")) or ""
        )
        if left_target != right_target:
            raise ValueError("Compared artifacts must have the same target_column")

        left_features = [
            str(feature_name)
            for feature_name in (left_artifact.get("selected_feature_columns") or [])
        ]
        right_features = [
            str(feature_name)
            for feature_name in (right_artifact.get("selected_feature_columns") or [])
        ]
        if (
            len(left_features) != len(right_features)
            or set(left_features) != set(right_features)
        ):
            raise ValueError(
                "Compared artifacts must have identical selected_feature_columns"
            )

        left_shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (left_artifact.get("shape_functions") or [])
        }
        right_shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (right_artifact.get("shape_functions") or [])
        }
        for feature_name in left_features:
            left_shape = left_shape_map.get(feature_name) or {}
            right_shape = right_shape_map.get(feature_name) or {}
            left_feature_type = str(
                left_shape.get("feature_type", "")
            )
            right_feature_type = str(
                right_shape.get("feature_type", "")
            )
            if left_feature_type != right_feature_type:
                raise ValueError(
                    f"Compared artifacts must render feature '{feature_name}' with the same chart type"
                )
            left_x_values = list(left_shape.get("x_values") or [])
            right_x_values = list(right_shape.get("x_values") or [])
            if left_feature_type == "categorical":
                if len(left_x_values) != len(right_x_values):
                    raise ValueError(
                        f"Compared artifacts must use aligned x_values for categorical feature '{feature_name}'"
                    )
                if left_x_values != right_x_values:
                    raise ValueError(
                        f"Compared artifacts must use the same x_values order for categorical feature '{feature_name}'"
                    )

        return left_features

    def _predict_model_compare_artifact_rows(
        self,
        artifact: Dict[str, Any],
        feature_rows: List[Dict[str, Any]],
        *,
        offsets: Optional[Dict[str, Dict[Any, float]]] = None,
        feature_shifts: Optional[Dict[str, float]] = None,
        intercept_adjustment: float = 0.0,
        exclude_missing_indicators: bool = False,
    ) -> np.ndarray:
        """Reconstruct predictions from one prepared artifact's shape functions."""
        missing_indicator_features = (
            self._compare_missing_indicator_features(artifact)
            if exclude_missing_indicators
            else set()
        )
        shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (artifact.get("shape_functions") or [])
        }
        categorical_features = {
            str(feature_name) for feature_name in (artifact.get("cat_features") or [])
        }
        predictions: List[float] = []
        for row in feature_rows:
            prediction = (
                float(artifact.get("base_prediction_offset", 0.0) or 0.0)
                + intercept_adjustment
            )
            for feature_name in artifact.get("feature_names") or []:
                if feature_name in missing_indicator_features:
                    continue
                shape_function = shape_map.get(str(feature_name))
                if shape_function is None:
                    continue
                value = row.get(feature_name)
                if str(shape_function.get("feature_type")) == "categorical":
                    prediction += ImportedIGANNRuntime._predict_categorical(
                        shape_function,
                        value,
                    )
                else:
                    prediction += ImportedIGANNRuntime._predict_numeric(
                        shape_function,
                        value,
                    )
                prediction += self._get_compare_offset_for_feature_value(
                    shape_function=shape_function,
                    original_is_categorical=feature_name in categorical_features,
                    value=value,
                    offsets=(offsets or {}).get(feature_name, {}),
                )
                prediction += float((feature_shifts or {}).get(feature_name, 0.0))
            predictions.append(float(prediction))
        return np.asarray(predictions, dtype=float)

    @staticmethod
    def _project_compare_rows(
        feature_rows: List[Dict[str, Any]],
        source_artifact: Dict[str, Any],
        target_artifact: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Apply the target export's fitted fills to shared preprocessed rows."""
        source_indicators = source_artifact.get("missing_indicator_map") or {}
        target_indicators = target_artifact.get("missing_indicator_map") or {}
        target_fills = target_artifact.get("numeric_missing_placeholder_values") or {}
        result = []
        for source_row in feature_rows:
            row = dict(source_row)
            for feature_name, target_indicator in target_indicators.items():
                source_indicator = source_indicators.get(feature_name)
                row[target_indicator] = (
                    MISSING_CATEGORY_VALUE
                    if source_indicator is not None
                    and source_row.get(source_indicator) == MISSING_CATEGORY_VALUE
                    else "Observed"
                )
            for feature_name, source_indicator in source_indicators.items():
                if (
                    source_row.get(source_indicator) == MISSING_CATEGORY_VALUE
                    and feature_name in target_fills
                ):
                    row[feature_name] = target_fills[feature_name]
            result.append({
                name: row.get(name)
                for name in target_artifact.get("feature_names") or []
            })
        return result

    def _build_held_out_test_performance(
        self,
        *,
        original_artifact: Dict[str, Any],
        edited_artifact: Dict[str, Any],
        test_data_artifact: Dict[str, Any],
        test_data_source: str,
        edited_offsets: Dict[str, Dict[Any, float]],
        edited_feature_shifts: Dict[str, float],
        intercept_adjustment: float = 0.0,
        intercept_correction_source: str = "manual",
    ) -> Optional[Dict[str, Any]]:
        """Evaluate full, reduced, and edited models on one shared row set."""
        held_out_test_data = test_data_artifact.get("held_out_test_data") or {}
        feature_rows = held_out_test_data.get("feature_rows") or []
        target_values = held_out_test_data.get("target_values") or []
        if not feature_rows or len(feature_rows) != len(target_values):
            return None

        actual_values = np.asarray(target_values, dtype=float)
        test_sample_ids = held_out_test_data.get("sample_ids") or []
        test_namespace = test_data_artifact.get("sample_id_namespace")
        def verify_test_exclusion(artifact: Dict[str, Any], label: str) -> bool:
            train_ids = artifact.get("training_sample_ids") or []
            if not (test_sample_ids and test_namespace and train_ids
                    and artifact.get("sample_id_namespace") == test_namespace):
                return False
            overlap = set(test_sample_ids).intersection(train_ids)
            if overlap:
                raise ValueError(
                    f"{label} training data contains {len(overlap)} selected test sample IDs"
                )
            return True
        full_exclusion_verified = verify_test_exclusion(original_artifact, "Full-data")
        reduced_exclusion_verified = verify_test_exclusion(edited_artifact, "Reduced-data")
        def rows_for(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Recreate model-specific missing placeholders from shared row semantics."""
            if artifact is test_data_artifact or (
                test_data_source == "main_training" and artifact is original_artifact
            ):
                return [
                    {name: source_row.get(name) for name in artifact.get("feature_names") or []}
                    for source_row in feature_rows
                ]
            return self._project_compare_rows(
                feature_rows, test_data_artifact, artifact,
            )
        full_predictions = self._predict_model_compare_artifact_rows(
            original_artifact,
            rows_for(original_artifact),
        )
        reduced_predictions = self._predict_model_compare_artifact_rows(
            edited_artifact,
            rows_for(edited_artifact),
        )
        edited_predictions = self._predict_model_compare_artifact_rows(
            edited_artifact,
            rows_for(edited_artifact),
            offsets=edited_offsets,
            feature_shifts=edited_feature_shifts,
            intercept_adjustment=intercept_adjustment,
        )
        def evaluate(predictions: np.ndarray) -> Dict[str, Optional[float]]:
            # Positive bias means overprediction; negative bias means underprediction.
            residuals = predictions - actual_values
            return {
                "mae": float(mean_absolute_error(actual_values, predictions)),
                "rmse": float(root_mean_squared_error(actual_values, predictions)),
                # Population SD of predicted values (ddof=0).
                "std_dev": float(np.std(predictions)),
                "mean_error": float(np.mean(residuals)),
                "r2": (
                    float(r2_score(actual_values, predictions))
                    if len(actual_values) >= 2 and np.ptp(actual_values) > 0
                    else None
                ),
            }

        error_change = np.abs(edited_predictions - actual_values) - np.abs(
            reduced_predictions - actual_values
        )
        rng = np.random.default_rng(42)
        bootstrap_differences = {key: [] for key in ("mae", "rmse", "r2")}
        for _ in range(1000):
            sample = rng.integers(0, len(actual_values), len(actual_values))
            sampled_actual = actual_values[sample]
            sampled_reduced = reduced_predictions[sample]
            sampled_edited = edited_predictions[sample]
            bootstrap_differences["mae"].append(
                float(np.mean(np.abs(sampled_edited - sampled_actual))
                      - np.mean(np.abs(sampled_reduced - sampled_actual)))
            )
            bootstrap_differences["rmse"].append(
                float(np.sqrt(np.mean((sampled_edited - sampled_actual) ** 2))
                      - np.sqrt(np.mean((sampled_reduced - sampled_actual) ** 2)))
            )
            if len(sampled_actual) >= 2 and np.ptp(sampled_actual) > 0:
                bootstrap_differences["r2"].append(
                    float(r2_score(sampled_actual, sampled_edited)
                          - r2_score(sampled_actual, sampled_reduced))
                )
        bootstrap_ci = {
            key: (
                [float(value) for value in np.percentile(samples, [2.5, 97.5])]
                if samples else None
            )
            for key, samples in bootstrap_differences.items()
        }

        def matching_training_rows(artifact: Dict[str, Any]) -> int:
            # Feature equality is only a warning signal; distinct people can share values.
            training = artifact.get("training_feature_rows") or []
            signatures = {json.dumps(row, sort_keys=True, default=str) for row in training}
            return sum(
                json.dumps(row, sort_keys=True, default=str) in signatures
                for row in rows_for(artifact)
            )

        reduced_test_data = edited_artifact.get("held_out_test_data") or {}
        same_exported_test_rows = (
            reduced_test_data.get("feature_rows") == feature_rows
            and reduced_test_data.get("target_values") == target_values
        ) if reduced_test_data.get("feature_rows") else None

        return {
            "test_size": len(feature_rows),
            "test_data_source": test_data_source,
            "baseline": evaluate(full_predictions),
            "reduced": evaluate(reduced_predictions),
            "edited": evaluate(edited_predictions),
            "edited_before_intercept_correction": (
                evaluate(edited_predictions - intercept_adjustment)
                if intercept_adjustment else None
            ),
            "actual_values": [float(value) for value in actual_values.tolist()],
            "baseline_predictions": [
                float(value) for value in full_predictions.tolist()
            ],
            "reduced_predictions": [float(value) for value in reduced_predictions.tolist()],
            "edited_predictions": [float(value) for value in edited_predictions.tolist()],
            "edited_before_intercept_predictions": (
                [float(value) for value in (edited_predictions - intercept_adjustment).tolist()]
                if intercept_adjustment else None
            ),
            "edit_effect": {
                "improved_percent": float(100 * np.mean(error_change < 0)),
                "worsened_percent": float(100 * np.mean(error_change > 0)),
                "unchanged_percent": float(100 * np.mean(error_change == 0)),
                "median_absolute_error_change": float(np.median(error_change)),
                "absolute_error_changes": [float(value) for value in error_change.tolist()],
            },
            "paired_bootstrap": {"resamples": 1000, "seed": 42, "ci_95": bootstrap_ci},
            "validation": {
                "same_exported_test_rows": same_exported_test_rows,
                "full_training_feature_matches": matching_training_rows(original_artifact),
                "reduced_training_feature_matches": matching_training_rows(edited_artifact),
                "full_training_rows_available": bool(original_artifact.get("training_feature_rows")),
                "reduced_training_rows_available": bool(edited_artifact.get("training_feature_rows")),
                "sample_ids_available": bool(test_sample_ids),
                "full_test_exclusion_verified": full_exclusion_verified,
                "reduced_test_exclusion_verified": reduced_exclusion_verified,
                "common_holdout_verified": (
                    full_exclusion_verified and reduced_exclusion_verified
                ),
                "preprocessing_scope": {
                    "full": original_artifact.get("preprocessing_scope") or "unknown",
                    "reduced": edited_artifact.get("preprocessing_scope") or "unknown",
                },
            },
            "intercept_correction": {
                "enabled": intercept_correction_source == "reduced training rows" or bool(intercept_adjustment),
                "shift": float(intercept_adjustment),
                "calculated_on": intercept_correction_source,
            },
        }

    def _main_training_performance_inputs(
        self,
        original_artifact: Dict[str, Any],
        edited_artifact: Dict[str, Any],
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """Check the active main-page training split for debug evaluation."""
        if (
            not self.is_trained
            or self.model_source != "trained"
            or self.X_train is None
            or self.X_train.empty
            or self.y_train is None
        ):
            return None
        feature_names = list(dict.fromkeys(
            (original_artifact.get("feature_names") or [])
            + (edited_artifact.get("feature_names") or [])
        ))
        if not set(feature_names).issubset(self.X_train.columns):
            return None
        targets = np.asarray(self.y_train, dtype=float).reshape(-1)
        if len(targets) != len(self.X_train) or not np.all(np.isfinite(targets)):
            return None
        return feature_names, targets

    @staticmethod
    def _compute_compare_mae(
        baseline_y_values: List[float],
        effective_y_values: List[float],
    ) -> float:
        if len(baseline_y_values) != len(effective_y_values):
            raise ValueError("Cannot compute compare MAE for misaligned feature values")
        if not baseline_y_values:
            return 0.0
        return float(mean_absolute_error(baseline_y_values, effective_y_values))

    @staticmethod
    def _compute_compare_rmse(
        baseline_y_values: List[float],
        effective_y_values: List[float],
    ) -> float:
        if len(baseline_y_values) != len(effective_y_values):
            raise ValueError("Cannot compute compare RMSE for misaligned feature values")
        if not baseline_y_values:
            return 0.0
        return float(root_mean_squared_error(baseline_y_values, effective_y_values))

    @staticmethod
    def _compute_compare_r2(
        baseline_y_values: List[float],
        effective_y_values: List[float],
    ) -> float:
        if len(baseline_y_values) != len(effective_y_values):
            raise ValueError("Cannot compute compare R2 for misaligned feature values")
        if len(baseline_y_values) < 2:
            return 1.0
        return float(r2_score(baseline_y_values, effective_y_values))

    @staticmethod
    def _interpolate_compare_values(
        source_x_values: List[Any],
        source_y_values: List[float],
        target_x_values: List[Any],
    ) -> List[float]:
        if (
            len(source_x_values) != len(source_y_values)
            or not source_x_values
            or not target_x_values
        ):
            return []

        try:
            source_x = np.asarray(source_x_values, dtype=float)
            source_y = np.asarray(source_y_values, dtype=float)
            target_x = np.asarray(target_x_values, dtype=float)
        except (TypeError, ValueError):
            return []

        if (
            source_x.size == 0
            or source_y.size == 0
            or source_x.size != source_y.size
            or target_x.size == 0
        ):
            return []

        order = np.argsort(source_x)
        source_x = source_x[order]
        source_y = source_y[order]

        return [
            float(value)
            for value in np.interp(target_x, source_x, source_y).tolist()
        ]

    @classmethod
    def _compute_numeric_compare_mae_on_reference_grid(
        cls,
        *,
        reference_x_values: List[Any],
        reference_y_values: List[float],
        comparison_x_values: List[Any],
        comparison_y_values: List[float],
    ) -> float:
        if (
            len(reference_x_values) != len(reference_y_values)
            or len(comparison_x_values) != len(comparison_y_values)
            or not reference_x_values
            or not comparison_x_values
        ):
            raise ValueError("Cannot compute compare MAE for invalid numeric feature values")

        try:
            comparison_numeric_x = np.asarray(comparison_x_values, dtype=float)
            reference_numeric_x = np.asarray(reference_x_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Cannot compute compare MAE for non-numeric reference grid values"
            ) from exc

        if comparison_numeric_x.size == 0 or reference_numeric_x.size == 0:
            raise ValueError("Cannot compute compare MAE without numeric feature values")

        domain_min = float(np.min(comparison_numeric_x))
        domain_max = float(np.max(comparison_numeric_x))
        overlap_mask = (
            (reference_numeric_x >= (domain_min - 1e-9))
            & (reference_numeric_x <= (domain_max + 1e-9))
        )
        if not np.any(overlap_mask):
            raise ValueError("Cannot compute compare MAE without overlapping numeric domain")

        filtered_reference_x = reference_numeric_x[overlap_mask].tolist()
        filtered_reference_y = [
            float(value)
            for include, value in zip(overlap_mask.tolist(), reference_y_values)
            if include
        ]
        interpolated_comparison_y = cls._interpolate_compare_values(
            comparison_x_values,
            comparison_y_values,
            filtered_reference_x,
        )
        return cls._compute_compare_mae(
            filtered_reference_y,
            interpolated_comparison_y,
        )

    @classmethod
    def _compute_numeric_compare_metric_on_reference_grid(
        cls,
        *,
        reference_x_values: List[Any],
        reference_y_values: List[float],
        comparison_x_values: List[Any],
        comparison_y_values: List[float],
        metric_name: str,
    ) -> float:
        if (
            len(reference_x_values) != len(reference_y_values)
            or len(comparison_x_values) != len(comparison_y_values)
            or not reference_x_values
            or not comparison_x_values
        ):
            raise ValueError(
                f"Cannot compute compare {metric_name} for invalid numeric feature values"
            )

        try:
            comparison_numeric_x = np.asarray(comparison_x_values, dtype=float)
            reference_numeric_x = np.asarray(reference_x_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot compute compare {metric_name} for non-numeric reference grid values"
            ) from exc

        if comparison_numeric_x.size == 0 or reference_numeric_x.size == 0:
            raise ValueError(
                f"Cannot compute compare {metric_name} without numeric feature values"
            )

        domain_min = float(np.min(comparison_numeric_x))
        domain_max = float(np.max(comparison_numeric_x))
        overlap_mask = (
            (reference_numeric_x >= (domain_min - 1e-9))
            & (reference_numeric_x <= (domain_max + 1e-9))
        )
        if not np.any(overlap_mask):
            raise ValueError(
                f"Cannot compute compare {metric_name} without overlapping numeric domain"
            )

        filtered_reference_x = reference_numeric_x[overlap_mask].tolist()
        filtered_reference_y = [
            float(value)
            for include, value in zip(overlap_mask.tolist(), reference_y_values)
            if include
        ]
        interpolated_comparison_y = cls._interpolate_compare_values(
            comparison_x_values,
            comparison_y_values,
            filtered_reference_x,
        )
        if metric_name == "MAE":
            return cls._compute_compare_mae(
                filtered_reference_y,
                interpolated_comparison_y,
            )
        if metric_name == "RMSE":
            return cls._compute_compare_rmse(
                filtered_reference_y,
                interpolated_comparison_y,
            )
        if metric_name == "R2":
            return cls._compute_compare_r2(
                filtered_reference_y,
                interpolated_comparison_y,
            )
        raise ValueError(f"Unsupported compare metric '{metric_name}'")

    @staticmethod
    def _compare_mae_status(
        current_mae: float,
        baseline_mae: float,
    ) -> str:
        if abs(current_mae - baseline_mae) < 0.000001:
            return "unchanged"
        if current_mae < baseline_mae:
            return "decreased"
        return "increased"

    @staticmethod
    def _compare_r2_status(
        current_r2: float,
        baseline_r2: float,
    ) -> str:
        if abs(current_r2 - baseline_r2) < 0.000001:
            return "unchanged"
        if current_r2 > baseline_r2:
            return "increased"
        return "decreased"

    @staticmethod
    def _collect_compare_offsets(
        prepared_artifact: Dict[str, Any],
        selected_submission_ids: Optional[List[str]],
        *,
        use_confidence: bool,
    ) -> Dict[str, Dict[Any, float]]:
        selected_set = (
            {str(submission_id) for submission_id in selected_submission_ids}
            if selected_submission_ids is not None
            else None
        )
        aggregate: Dict[str, Dict[Any, Dict[str, float]]] = {}

        for feature_group in prepared_artifact.get("submissions_by_feature") or []:
            feature_name = str(feature_group.get("feature_name", "")).strip()
            if not feature_name:
                continue
            feature_bucket = aggregate.setdefault(feature_name, {})
            for submission in feature_group.get("submissions") or []:
                submission_id = str(submission.get("submission_id", "")).strip()
                if selected_set is not None and submission_id not in selected_set:
                    continue
                for point in submission.get("edited_points") or []:
                    raw_x_value = point.get("x_value")
                    point_bucket = feature_bucket.setdefault(
                        raw_x_value,
                        {"offset_sum": 0.0, "point_count": 0.0},
                    )
                    contribution = float(point.get("y_value", 0.0))
                    if use_confidence:
                        contribution *= float(point.get("weight", 0.5))
                    point_bucket["offset_sum"] += contribution
                    point_bucket["point_count"] += 1.0

        combined: Dict[str, Dict[Any, float]] = {}
        for feature_name, point_map in aggregate.items():
            combined[feature_name] = {}
            for raw_x_value, stats in point_map.items():
                point_count = float(stats.get("point_count", 0.0) or 0.0)
                combined[feature_name][raw_x_value] = (
                    float(stats.get("offset_sum", 0.0)) / point_count
                    if point_count > 0
                    else 0.0
                )
        return combined

    @staticmethod
    def _collect_compare_submission_offsets(
        prepared_artifact: Dict[str, Any],
        selected_submission_ids: Optional[List[str]],
        *,
        use_confidence: bool,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return one offset map per selected submission, grouped by feature.

        The aggregate preview averages all selected point contributions.  The
        individual-trace preview deliberately keeps each submission separate,
        while applying the same confidence weighting to its points.
        """
        selected_set = (
            {str(submission_id) for submission_id in selected_submission_ids}
            if selected_submission_ids is not None
            else None
        )
        traces: Dict[str, List[Dict[str, Any]]] = {}

        for feature_group in prepared_artifact.get("submissions_by_feature") or []:
            feature_name = str(feature_group.get("feature_name", "")).strip()
            if not feature_name:
                continue
            for submission in feature_group.get("submissions") or []:
                submission_id = str(submission.get("submission_id", "")).strip()
                if selected_set is not None and submission_id not in selected_set:
                    continue

                point_stats: Dict[Any, Dict[str, float]] = {}
                for point in submission.get("edited_points") or []:
                    raw_x_value = point.get("x_value")
                    stats = point_stats.setdefault(
                        raw_x_value,
                        {"offset_sum": 0.0, "point_count": 0.0},
                    )
                    contribution = float(point.get("y_value", 0.0))
                    if use_confidence:
                        contribution *= float(point.get("weight", 0.5))
                    stats["offset_sum"] += contribution
                    stats["point_count"] += 1.0

                offsets = {
                    raw_x_value: float(stats["offset_sum"])
                    / float(stats["point_count"])
                    for raw_x_value, stats in point_stats.items()
                    if float(stats["point_count"]) > 0
                }
                traces.setdefault(feature_name, []).append(
                    {
                        "submission_id": submission_id,
                        "user_name": str(submission.get("user_name", "") or ""),
                        "sureness": int(submission.get("sureness", 0) or 0),
                        "offsets": offsets,
                    }
                )
        return traces

    @staticmethod
    def _collect_compare_expert_offsets(
        prepared_artifact: Dict[str, Any],
        selected_submission_ids: Optional[List[str]],
        *,
        use_confidence: bool,
    ) -> Dict[str, Dict[str, Any]]:
        """Combine each selected expert's edits across all of their features."""
        selected_set = (
            {str(submission_id) for submission_id in selected_submission_ids}
            if selected_submission_ids is not None
            else None
        )
        experts: Dict[str, Dict[str, Any]] = {}

        for feature_group in prepared_artifact.get("submissions_by_feature") or []:
            feature_name = str(feature_group.get("feature_name", "")).strip()
            if not feature_name:
                continue
            for submission in feature_group.get("submissions") or []:
                submission_id = str(submission.get("submission_id", "")).strip()
                if selected_set is not None and submission_id not in selected_set:
                    continue
                user_name = str(submission.get("user_name", "") or "").strip()
                expert_key = user_name or submission_id
                if not expert_key:
                    continue
                expert = experts.setdefault(
                    expert_key,
                    {
                        "user_name": user_name or submission_id,
                        "submission_ids": set(),
                        "offset_stats": {},
                    },
                )
                expert["submission_ids"].add(submission_id)
                point_stats = expert["offset_stats"].setdefault(feature_name, {})
                for point in submission.get("edited_points") or []:
                    raw_x_value = point.get("x_value")
                    stats = point_stats.setdefault(
                        raw_x_value,
                        {"offset_sum": 0.0, "point_count": 0.0},
                    )
                    contribution = float(point.get("y_value", 0.0))
                    if use_confidence:
                        contribution *= float(point.get("weight", 0.5))
                    stats["offset_sum"] += contribution
                    stats["point_count"] += 1.0

        for expert in experts.values():
            expert["offsets"] = {
                feature_name: {
                    raw_x_value: float(stats["offset_sum"])
                    / float(stats["point_count"])
                    for raw_x_value, stats in point_stats.items()
                    if float(stats["point_count"]) > 0
                }
                for feature_name, point_stats in expert["offset_stats"].items()
            }
        return experts

    @classmethod
    def _compute_compare_average_prediction_score(
        cls,
        prepared_artifact: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
    ) -> float:
        """Estimate an artifact's mean prediction across its displayed grids.

        Each feature curve is sampled uniformly over the same x-values shown on
        the comparison page. Their means are additive in the exported IGANN
        representation, together with the artifact's base prediction offset.
        """
        contributions = cls._compute_compare_feature_average_contributions(
            prepared_artifact,
            offsets,
            method="uniform",
        )
        return float(prepared_artifact.get("base_prediction_offset", 0.0) or 0.0) + sum(
            contributions.values()
        )

    @classmethod
    def _compute_compare_data_weighted_average_prediction_score(
        cls,
        prepared_artifact: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
    ) -> Optional[float]:
        """Estimate an average using the artifact's feature distributions.

        Unlike the uniform measure, each contribution is weighted by the
        observed frequency of that feature value (or numeric bin). This is a
        marginal shape-function estimate, not a reconstruction of joint rows.
        """
        contributions = cls._compute_compare_feature_average_contributions(
            prepared_artifact,
            offsets,
            method="data_weighted",
        )
        if contributions is None:
            return None
        return float(prepared_artifact.get("base_prediction_offset", 0.0) or 0.0) + sum(
            contributions.values()
        )

    @staticmethod
    def _compare_missing_indicator_features(*artifacts: Dict[str, Any]) -> set[str]:
        """Identify generated missingness features in compared artifacts."""
        excluded = set()
        for artifact in artifacts:
            excluded.update(
                str(name)
                for name in (artifact.get("missing_indicator_map") or {}).values()
            )
            excluded.update(
                str(name)
                for name in (artifact.get("feature_names") or [])
                if str(name).endswith(MISSING_INDICATOR_SUFFIX)
            )
        return excluded

    @classmethod
    def _compute_compare_feature_average_contributions(
        cls,
        prepared_artifact: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
        *,
        method: str,
    ) -> Optional[Dict[str, float]]:
        """Return one averaging-method contribution per public shape function."""
        categorical_features = {
            str(feature_name)
            for feature_name in (prepared_artifact.get("cat_features") or [])
        }
        contributions: Dict[str, float] = {}

        for shape_function in prepared_artifact.get("shape_functions") or []:
            feature_name = str(shape_function.get("feature_name", "")).strip()
            if not feature_name:
                continue

            if method == "uniform":
                x_values = list(shape_function.get("x_values") or [])
                y_values = shape_function.get("y_values") or []
                if len(x_values) != len(y_values) or not x_values:
                    continue
                weighted_values = [(value, 1) for value in x_values]
            elif method == "data_weighted":
                distribution = shape_function.get("distribution") or {}
                if not distribution:
                    return None
                if distribution.get("chart_type") == "categorical":
                    weighted_values = [
                        (item.get("x_value"), int(item.get("count", 0) or 0))
                        for item in (distribution.get("counts") or [])
                        if int(item.get("count", 0) or 0) > 0
                    ]
                elif distribution.get("chart_type") == "numeric":
                    weighted_values = [
                        (item.get("center"), int(item.get("count", 0) or 0))
                        for item in (distribution.get("bins") or [])
                        if int(item.get("count", 0) or 0) > 0
                    ]
                else:
                    return None
            else:
                raise ValueError(f"Unknown average contribution method '{method}'")

            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight <= 0:
                return None

            contribution = 0.0
            for value, weight in weighted_values:
                if shape_function.get("feature_type") == "categorical":
                    base_value = cls._predict_categorical_shape_value(
                        shape_function,
                        value,
                    )
                else:
                    base_value = cls._predict_numeric_shape_value(
                        shape_function,
                        value,
                    )
                offset = cls._get_compare_offset_for_feature_value(
                    shape_function=shape_function,
                    original_is_categorical=feature_name in categorical_features,
                    value=value,
                    offsets=offsets.get(feature_name, {}),
                )
                contribution += weight * (base_value + float(offset))
            contributions[feature_name] = contribution / total_weight

        return contributions

    @classmethod
    def _compute_compare_feature_average_contributions_on_reference(
        cls,
        *,
        reference_artifact: Dict[str, Any],
        evaluated_artifact: Dict[str, Any],
        evaluated_offsets: Dict[str, Dict[Any, float]],
        feature_shifts: Optional[Dict[str, float]] = None,
        method: str,
        numeric_grid_artifact: Optional[Dict[str, Any]] = None,
        sample_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, float]]:
        """Evaluate both artifacts on the original artifact's feature support.

        This supplies like-for-like feature contributions for the model
        comparison's average-prediction section.  The reference artifact is
        the full-data/original model; the evaluated artifact may be that model
        itself or the edited model with selected edit offsets applied. The
        test-set method samples the exact held-out rows in the original JSON.
        The uniform method samples numeric curves on the shared 100-point
        metric grid over the two artifacts' overlapping domain.
        """
        row_based_method = method in {"test_set", "main_dataset"}
        if method == "test_set":
            sample_rows = (
                (reference_artifact.get("held_out_test_data") or {}).get("feature_rows")
                or []
            )
        if row_based_method and not sample_rows:
            return None
        missing_indicator_features = cls._compare_missing_indicator_features(
            reference_artifact, evaluated_artifact
        )
        public_features = set(reference_artifact.get("selected_feature_columns") or [])
        reference_shapes = [
            shape_function
            for shape_function in (reference_artifact.get("shape_functions") or [])
            if shape_function.get("feature_name") in public_features
            and shape_function.get("feature_name") not in missing_indicator_features
        ]
        if row_based_method:
            feature_names = dict.fromkeys(
                (reference_artifact.get("feature_names") or [])
                + (evaluated_artifact.get("feature_names") or [])
            )
            reference_shapes = [
                {"feature_name": feature_name}
                for feature_name in feature_names
                if feature_name not in missing_indicator_features
            ]
        evaluated_shape_map = {
            str(shape_function.get("feature_name", "")).strip(): shape_function
            for shape_function in (evaluated_artifact.get("shape_functions") or [])
        }
        numeric_grid_shape_map = {
            str(shape_function.get("feature_name", "")).strip(): shape_function
            for shape_function in (
                (numeric_grid_artifact or evaluated_artifact).get("shape_functions") or []
            )
        }
        evaluated_categorical_features = {
            str(feature_name)
            for feature_name in (evaluated_artifact.get("cat_features") or [])
        }
        contributions: Dict[str, float] = {}

        for reference_shape in reference_shapes:
            feature_name = str(reference_shape.get("feature_name", "")).strip()
            evaluated_shape = evaluated_shape_map.get(feature_name)
            if not feature_name:
                return None
            if not evaluated_shape:
                if row_based_method:
                    contributions[feature_name] = 0.0
                    continue
                return None

            if method == "uniform":
                reference_x_values = list(reference_shape.get("x_values") or [])
                reference_y_values = list(reference_shape.get("y_values") or [])
                if (
                    len(reference_x_values) != len(reference_y_values)
                    or not reference_x_values
                ):
                    return None
                if reference_shape.get("feature_type") == "numeric":
                    grid_shape = numeric_grid_shape_map.get(feature_name)
                    grid_x_values = list((grid_shape or {}).get("x_values") or [])
                    if not grid_x_values:
                        return None
                    reference_numeric_x = np.asarray(reference_x_values, dtype=float)
                    grid_numeric_x = np.asarray(grid_x_values, dtype=float)
                    overlap_min = max(
                        float(np.min(reference_numeric_x)),
                        float(np.min(grid_numeric_x)),
                    )
                    overlap_max = min(
                        float(np.max(reference_numeric_x)),
                        float(np.max(grid_numeric_x)),
                    )
                    if overlap_min > overlap_max:
                        return None
                    weighted_values = [
                        (value, 1)
                        for value in np.linspace(
                            overlap_min,
                            overlap_max,
                            MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS,
                        )
                    ]
                else:
                    weighted_values = [(value, 1) for value in reference_x_values]
            elif method == "data_weighted":
                distribution = reference_shape.get("distribution") or {}
                if distribution.get("chart_type") == "categorical":
                    weighted_values = [
                        (item.get("x_value"), int(item.get("count", 0) or 0))
                        for item in (distribution.get("counts") or [])
                        if int(item.get("count", 0) or 0) > 0
                    ]
                elif distribution.get("chart_type") == "numeric":
                    weighted_values = [
                        (item.get("center"), int(item.get("count", 0) or 0))
                        for item in (distribution.get("bins") or [])
                        if int(item.get("count", 0) or 0) > 0
                    ]
                else:
                    return None
            elif row_based_method:
                weighted_values = [
                    (row.get(feature_name), 1) for row in sample_rows
                ]
            else:
                raise ValueError(f"Unknown average contribution method '{method}'")

            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight <= 0:
                return None

            contribution = 0.0
            for value, weight in weighted_values:
                if evaluated_shape.get("feature_type") == "categorical":
                    base_value = cls._predict_categorical_shape_value(
                        evaluated_shape,
                        value,
                    )
                else:
                    base_value = cls._predict_numeric_shape_value(
                        evaluated_shape,
                        value,
                    )
                offset = cls._get_compare_offset_for_feature_value(
                    shape_function=evaluated_shape,
                    original_is_categorical=(
                        feature_name in evaluated_categorical_features
                    ),
                    value=value,
                    offsets=evaluated_offsets.get(feature_name, {}),
                )
                contribution += weight * (
                    base_value
                    + float(offset)
                    + float((feature_shifts or {}).get(feature_name, 0.0))
                )
            contributions[feature_name] = contribution / total_weight

        return contributions

    @staticmethod
    def _predict_numeric_shape_value(
        shape_function: Dict[str, Any],
        value: Any,
    ) -> float:
        try:
            x_values = np.asarray(shape_function.get("x_values") or [], dtype=float)
            y_values = np.asarray(shape_function.get("y_values") or [], dtype=float)
            return float(np.interp(float(value), x_values, y_values))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _predict_categorical_shape_value(
        shape_function: Dict[str, Any],
        value: Any,
    ) -> float:
        x_values = [str(item) for item in (shape_function.get("x_values") or [])]
        y_values = shape_function.get("y_values") or []
        try:
            return float(y_values[x_values.index(str(value))])
        except (ValueError, IndexError, TypeError):
            return 0.0

    @staticmethod
    def _scale_compare_offsets(
        offsets: Dict[str, Dict[Any, float]],
        scale_factor: float,
        *,
        feature_names: Optional[set[str]] = None,
    ) -> Dict[str, Dict[Any, float]]:
        """Scale selected edit offsets by one preview-only factor."""
        return {
            feature_name: {
                value: (
                    float(offset) * scale_factor
                    if feature_names is None or feature_name in feature_names
                    else float(offset)
                )
                for value, offset in point_offsets.items()
            }
            for feature_name, point_offsets in offsets.items()
        }

    @staticmethod
    def _scale_compare_feature_shifts(
        feature_shifts: Dict[str, float],
        scale_factor: float,
        *,
        feature_names: Optional[set[str]] = None,
    ) -> Dict[str, float]:
        """Scale preview-only whole-feature shifts for the selected features."""
        return {
            feature_name: (
                float(shift) * scale_factor
                if feature_names is None or feature_name in feature_names
                else float(shift)
            )
            for feature_name, shift in feature_shifts.items()
        }

    @classmethod
    def _build_overall_shape_function_mae(
        cls,
        per_feature_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarize full-data discrepancy across the compared shape functions.

        ``pooled`` weights a feature by the number of full-data reference-grid
        points used for its MAE. ``mean_per_feature`` gives every compared
        shape function the same weight, regardless of its displayed grid size.
        """
        valid_metrics = [
            metric
            for metric in per_feature_metrics
            if int(metric.get("point_count", 0) or 0) > 0
            and np.isfinite(metric.get("baseline", np.nan))
            and np.isfinite(metric.get("edited", np.nan))
        ]
        feature_count = len(valid_metrics)
        point_count = sum(int(metric["point_count"]) for metric in valid_metrics)

        def summary(baseline: float, edited: float) -> Dict[str, Any]:
            return {
                "baseline": baseline,
                "edited": edited,
                "status": cls._compare_mae_status(edited, baseline),
            }

        if not valid_metrics or point_count <= 0:
            return {
                "feature_count": 0,
                "point_count": 0,
                "pooled": summary(0.0, 0.0),
                "mean_per_feature": summary(0.0, 0.0),
            }

        pooled_baseline = sum(
            float(metric["baseline"]) * int(metric["point_count"])
            for metric in valid_metrics
        ) / point_count
        pooled_edited = sum(
            float(metric["edited"]) * int(metric["point_count"])
            for metric in valid_metrics
        ) / point_count
        mean_baseline = sum(
            float(metric["baseline"]) for metric in valid_metrics
        ) / feature_count
        mean_edited = sum(
            float(metric["edited"]) for metric in valid_metrics
        ) / feature_count
        return {
            "feature_count": feature_count,
            "point_count": point_count,
            "pooled": summary(pooled_baseline, pooled_edited),
            "mean_per_feature": summary(mean_baseline, mean_edited),
        }

    @classmethod
    def _build_expert_complete_model_mae(
        cls,
        per_expert_feature_metrics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Pool each expert's model MAE over every selected shape function."""
        summaries: Dict[str, Dict[str, Any]] = {}
        for metric in per_expert_feature_metrics:
            point_count = int(metric.get("point_count", 0) or 0)
            mae = float(metric.get("mae", np.nan))
            baseline_mae = float(metric.get("baseline_mae", np.nan))
            user_name = str(metric.get("user_name", "") or "").strip()
            if (
                not user_name
                or point_count <= 0
                or not np.isfinite(mae)
                or not np.isfinite(baseline_mae)
            ):
                continue
            summary = summaries.setdefault(
                user_name,
                {
                    "user_name": user_name,
                    "mae_sum": 0.0,
                    "baseline_mae_sum": 0.0,
                    "feature_count": 0,
                    "point_count": 0,
                    "submission_ids": set(),
                },
            )
            summary["mae_sum"] += mae * point_count
            summary["baseline_mae_sum"] += baseline_mae * point_count
            summary["feature_count"] += 1
            summary["point_count"] += point_count
            summary["submission_ids"].update(metric.get("submission_ids", set()))

        return [
            {
                "user_name": summary["user_name"],
                "mae": summary["mae_sum"] / summary["point_count"],
                "baseline_mae": summary["baseline_mae_sum"]
                / summary["point_count"],
                "status": cls._compare_mae_status(
                    summary["mae_sum"] / summary["point_count"],
                    summary["baseline_mae_sum"] / summary["point_count"],
                ),
                "feature_count": summary["feature_count"],
                "point_count": summary["point_count"],
                "submission_count": len(summary["submission_ids"]),
            }
            for summary in sorted(
                summaries.values(),
                key=lambda summary: summary["user_name"].casefold(),
            )
            if summary["point_count"] > 0
        ]

    def _build_all_feature_mae_change(
        self,
        *,
        left_artifact: Dict[str, Any],
        right_artifact: Dict[str, Any],
        shared_features: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compare all exported edits for each feature in both confidence modes."""
        left_shapes = {
            str(shape.get("feature_name")): shape
            for shape in left_artifact.get("shape_functions") or []
        }
        right_shapes = {
            str(shape.get("feature_name")): shape
            for shape in right_artifact.get("shape_functions") or []
        }
        right_categorical = set(right_artifact.get("cat_features") or [])
        results: Dict[str, List[Dict[str, Any]]] = {}

        for mode, use_confidence in (
            ("without_confidence", False),
            ("with_confidence", True),
        ):
            offsets = self._collect_compare_offsets(
                right_artifact, None, use_confidence=use_confidence
            )
            feature_metrics = []
            for feature_name in shared_features:
                left_shape = left_shapes.get(feature_name)
                right_shape = right_shapes.get(feature_name)
                if not left_shape or not right_shape:
                    continue
                left_x = list(left_shape.get("x_values") or [])
                right_x = list(right_shape.get("x_values") or [])
                left_y = [float(value) for value in left_shape.get("y_values") or []]
                right_y = [float(value) for value in right_shape.get("y_values") or []]
                if (
                    not left_x
                    or not right_x
                    or len(left_x) != len(left_y)
                    or len(right_x) != len(right_y)
                ):
                    continue

                edited_y = [
                    base_y
                    + self._get_compare_offset_for_feature_value(
                        shape_function=right_shape,
                        original_is_categorical=feature_name in right_categorical,
                        value=x_value,
                        offsets=offsets.get(feature_name, {}),
                    )
                    for x_value, base_y in zip(right_x, right_y)
                ]
                if str(left_shape.get("feature_type", "numeric")) == "numeric":
                    left_numeric_x = np.asarray(left_x, dtype=float)
                    right_numeric_x = np.asarray(right_x, dtype=float)
                    overlap_min = max(
                        float(np.min(left_numeric_x)), float(np.min(right_numeric_x))
                    )
                    overlap_max = min(
                        float(np.max(left_numeric_x)), float(np.max(right_numeric_x))
                    )
                    if overlap_min > overlap_max:
                        raise ValueError(
                            f"Cannot compute all-feature MAE change without overlapping numeric domain for '{feature_name}'"
                        )
                    sample_x = np.linspace(
                        overlap_min,
                        overlap_max,
                        MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS,
                    ).tolist()
                    reference_y = self._interpolate_compare_values(
                        left_x, left_y, sample_x
                    )
                    baseline_y = self._interpolate_compare_values(
                        right_x, right_y, sample_x
                    )
                    sampled_edited_y = self._interpolate_compare_values(
                        right_x, edited_y, sample_x
                    )
                else:
                    reference_y = left_y
                    baseline_y = right_y
                    sampled_edited_y = edited_y
                baseline_mae = self._compute_compare_mae(reference_y, baseline_y)
                edited_mae = self._compute_compare_mae(reference_y, sampled_edited_y)
                feature_metrics.append(
                    {
                        "feature_name": feature_name,
                        "mae": edited_mae,
                        "baseline_mae": baseline_mae,
                        "status": self._compare_mae_status(edited_mae, baseline_mae),
                        "point_count": len(reference_y),
                    }
                )
            results[mode] = feature_metrics
        return results

    def build_model_compare_preview(
        self,
        *,
        left_artifact: Dict[str, Any],
        right_artifact: Dict[str, Any],
        left_selected_submission_ids: Optional[List[str]],
        right_selected_submission_ids: Optional[List[str]],
        use_confidence: bool = True,
        limit_edited_mean_deviation: bool = False,
        feature_names: Optional[List[str]] = None,
        overall_mae_feature_names: Optional[List[str]] = None,
        mean_deviation_feature_names: Optional[List[str]] = None,
        mean_deviation_scaled_feature_names: Optional[List[str]] = None,
        manual_feature_shifts: Optional[Dict[str, float]] = None,
        held_out_test_source: str = "right",
        intercept_adjustment: float = 0.0,
        apply_training_intercept_correction: bool = False,
        feature_metrics_on_original_test_set: bool = False,
    ) -> Dict[str, Any]:
        intercept_adjustment = float(intercept_adjustment)
        if not np.isfinite(intercept_adjustment):
            raise ValueError("intercept_adjustment must be finite")
        original_edited_base = float(right_artifact.get("base_prediction_offset", 0.0) or 0.0)
        adjusted_edited_base = original_edited_base + intercept_adjustment
        if not np.isfinite(adjusted_edited_base):
            raise ValueError("adjusted edited base prediction offset must be finite")
        held_out_test_source = str(held_out_test_source or "right").strip().lower()
        if held_out_test_source not in {"left", "right", "main_training"}:
            raise ValueError(
                "held_out_test_source must be 'left', 'right', or 'main_training'"
            )
        original_test_rows = (
            (left_artifact.get("held_out_test_data") or {}).get("feature_rows") or []
        )
        if feature_metrics_on_original_test_set and not original_test_rows:
            raise ValueError(
                "Original artifact has no held-out feature rows for per-feature metrics"
            )
        shared_features = self._validate_model_compare_pair(
            left_artifact,
            right_artifact,
        )
        if feature_names:
            requested = []
            shared_feature_set = set(shared_features)
            for feature_name in feature_names:
                normalized_name = str(feature_name).strip()
                if normalized_name and normalized_name in shared_feature_set:
                    requested.append(normalized_name)
            active_features = requested or shared_features
        else:
            active_features = shared_features
        overall_mae_feature_set = (
            {str(feature_name).strip() for feature_name in overall_mae_feature_names}
            if overall_mae_feature_names is not None
            else set(active_features)
        )
        mean_deviation_feature_set = (
            {
                str(feature_name).strip()
                for feature_name in mean_deviation_feature_names
                if str(feature_name).strip() in active_features
            }
            if mean_deviation_feature_names is not None
            else set(shared_features)
        )
        explicitly_scaled_mean_deviation_feature_set = (
            {
                str(feature_name).strip()
                for feature_name in mean_deviation_scaled_feature_names
                if str(feature_name).strip() in mean_deviation_feature_set
            }
            if mean_deviation_scaled_feature_names is not None
            else set()
        )
        mean_deviation_scaled_feature_set = (
            explicitly_scaled_mean_deviation_feature_set
            or mean_deviation_feature_set
        )
        should_limit_edited_mean_deviation = bool(
            limit_edited_mean_deviation
            or explicitly_scaled_mean_deviation_feature_set
        )
        right_model_feature_names = set(right_artifact.get("feature_names") or [])
        right_manual_feature_shifts = {
            str(feature_name).strip(): float(shift)
            for feature_name, shift in (manual_feature_shifts or {}).items()
            if str(feature_name).strip() in right_model_feature_names
            and np.isfinite(float(shift))
            and abs(float(shift)) > 1e-12
        }

        left_shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (left_artifact.get("shape_functions") or [])
        }
        right_shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (right_artifact.get("shape_functions") or [])
        }
        left_offsets = self._collect_compare_offsets(
            left_artifact,
            left_selected_submission_ids,
            use_confidence=use_confidence,
        )
        right_offsets = self._collect_compare_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=use_confidence,
        )
        alternate_confidence_offsets = self._collect_compare_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=not use_confidence,
        )
        left_submission_offsets = self._collect_compare_submission_offsets(
            left_artifact,
            left_selected_submission_ids,
            use_confidence=use_confidence,
        )
        right_submission_offsets = self._collect_compare_submission_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=use_confidence,
        )
        right_alternate_submission_offsets = self._collect_compare_submission_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=not use_confidence,
        )
        right_expert_offsets = self._collect_compare_expert_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=use_confidence,
        )
        left_cat_features = {
            str(feature_name)
            for feature_name in (left_artifact.get("cat_features") or [])
        }
        right_cat_features = {
            str(feature_name)
            for feature_name in (right_artifact.get("cat_features") or [])
        }
        def score_from_contributions(
            artifact: Dict[str, Any],
            contributions: Optional[Dict[str, float]],
            included_features: Optional[set[str]] = None,
        ) -> Optional[float]:
            if contributions is None:
                return None
            return float(artifact.get("base_prediction_offset", 0.0) or 0.0) + sum(
                value
                for feature_name, value in contributions.items()
                if included_features is None or feature_name in included_features
            )

        # The full-data artifact supplies the reference curve grid. Its base
        # curve is used without imported edits.
        original_uniform_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=left_artifact,
                evaluated_offsets={},
                method="uniform",
                numeric_grid_artifact=right_artifact,
            )
        )
        unconstrained_edited_uniform_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets=right_offsets,
                feature_shifts=right_manual_feature_shifts,
                method="uniform",
                numeric_grid_artifact=right_artifact,
            )
        )
        right_base_uniform_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets={},
                method="uniform",
                numeric_grid_artifact=right_artifact,
            )
        )
        original_average_prediction_score = score_from_contributions(
            left_artifact,
            original_uniform_contributions,
        )
        reduced_average_prediction_score = score_from_contributions(
            right_artifact,
            right_base_uniform_contributions,
        )
        mean_deviation_original_average = score_from_contributions(
            left_artifact,
            original_uniform_contributions,
            mean_deviation_feature_set,
        )
        unconstrained_mean_deviation_edited_average = score_from_contributions(
            right_artifact,
            unconstrained_edited_uniform_contributions,
            mean_deviation_feature_set,
        )
        right_base_mean_deviation_average = score_from_contributions(
            right_artifact,
            right_base_uniform_contributions,
            mean_deviation_feature_set,
        )
        unconstrained_scaled_feature_average = score_from_contributions(
            right_artifact,
            unconstrained_edited_uniform_contributions,
            mean_deviation_scaled_feature_set,
        )
        right_base_scaled_feature_average = score_from_contributions(
            right_artifact,
            right_base_uniform_contributions,
            mean_deviation_scaled_feature_set,
        )
        if (
            original_average_prediction_score is None
            or reduced_average_prediction_score is None
            or mean_deviation_original_average is None
            or unconstrained_mean_deviation_edited_average is None
            or right_base_mean_deviation_average is None
            or unconstrained_scaled_feature_average is None
            or right_base_scaled_feature_average is None
        ):
            raise ValueError("Unable to calculate average predictions on the full-data reference")
        mean_deviation_limit = 0.10 * abs(mean_deviation_original_average)
        scale_factor = 1.0
        right_effective_feature_shifts = dict(right_manual_feature_shifts)
        if (
            should_limit_edited_mean_deviation
            and abs(
                unconstrained_mean_deviation_edited_average
                - mean_deviation_original_average
            )
            > mean_deviation_limit
        ):
            lower_bound = mean_deviation_original_average - mean_deviation_limit
            upper_bound = mean_deviation_original_average + mean_deviation_limit
            target_average = min(
                max(unconstrained_mean_deviation_edited_average, lower_bound),
                upper_bound,
            )
            scaled_feature_edit_mean_effect = (
                unconstrained_scaled_feature_average
                - right_base_scaled_feature_average
            )
            if abs(scaled_feature_edit_mean_effect) > 1e-12:
                scale_factor = min(
                    1.0,
                    max(
                        0.0,
                        1.0
                        + (
                            target_average
                            - unconstrained_mean_deviation_edited_average
                        )
                        / scaled_feature_edit_mean_effect,
                    ),
                )
                right_offsets = self._scale_compare_offsets(
                    right_offsets,
                    scale_factor,
                    feature_names=mean_deviation_scaled_feature_set,
                )
                right_effective_feature_shifts = self._scale_compare_feature_shifts(
                    right_manual_feature_shifts,
                    scale_factor,
                    feature_names=mean_deviation_scaled_feature_set,
                )
                for feature_name, trace in right_submission_offsets.items():
                    if feature_name not in mean_deviation_scaled_feature_set:
                        continue
                    for submission_trace in trace:
                        submission_trace["offsets"] = {
                            value: float(offset) * scale_factor
                            for value, offset in submission_trace["offsets"].items()
                        }

        alternate_confidence_feature_shifts = dict(right_manual_feature_shifts)
        if should_limit_edited_mean_deviation:
            alternate_contributions = (
                self._compute_compare_feature_average_contributions_on_reference(
                    reference_artifact=left_artifact,
                    evaluated_artifact=right_artifact,
                    evaluated_offsets=alternate_confidence_offsets,
                    feature_shifts=right_manual_feature_shifts,
                    method="uniform",
                    numeric_grid_artifact=right_artifact,
                )
            )
            alternate_average = score_from_contributions(
                right_artifact,
                alternate_contributions,
                mean_deviation_feature_set,
            )
            alternate_scaled_average = score_from_contributions(
                right_artifact,
                alternate_contributions,
                mean_deviation_scaled_feature_set,
            )
            if alternate_average is None or alternate_scaled_average is None:
                raise ValueError("Unable to calculate alternate confidence average")
            if abs(alternate_average - mean_deviation_original_average) > mean_deviation_limit:
                lower_bound = mean_deviation_original_average - mean_deviation_limit
                upper_bound = mean_deviation_original_average + mean_deviation_limit
                target_average = min(max(alternate_average, lower_bound), upper_bound)
                scaled_edit_effect = (
                    alternate_scaled_average - right_base_scaled_feature_average
                )
                if abs(scaled_edit_effect) > 1e-12:
                    alternate_scale_factor = min(
                        1.0,
                        max(
                            0.0,
                            1.0 + (target_average - alternate_average) / scaled_edit_effect,
                        ),
                    )
                    alternate_confidence_offsets = self._scale_compare_offsets(
                        alternate_confidence_offsets,
                        alternate_scale_factor,
                        feature_names=mean_deviation_scaled_feature_set,
                    )
                    alternate_confidence_feature_shifts = self._scale_compare_feature_shifts(
                        right_manual_feature_shifts,
                        alternate_scale_factor,
                        feature_names=mean_deviation_scaled_feature_set,
                    )
        if apply_training_intercept_correction:
            training_rows = right_artifact.get("training_feature_rows") or []
            if not training_rows:
                raise ValueError(
                    "Training-only intercept correction requires reduced-model training rows"
                )
            original_training_predictions = self._predict_model_compare_artifact_rows(
                right_artifact, training_rows,
            )
            edited_training_predictions = self._predict_model_compare_artifact_rows(
                right_artifact, training_rows,
                offsets=right_offsets,
                feature_shifts=right_effective_feature_shifts,
            )
            intercept_adjustment = float(np.mean(
                original_training_predictions - edited_training_predictions
            ))
            adjusted_edited_base = original_edited_base + intercept_adjustment
        edited_uniform_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets=right_offsets,
                feature_shifts=right_effective_feature_shifts,
                method="uniform",
                numeric_grid_artifact=right_artifact,
            )
        )
        edited_average_prediction_score = score_from_contributions(
            right_artifact,
            edited_uniform_contributions,
        )
        constrained_mean_deviation_edited_average = score_from_contributions(
            right_artifact,
            edited_uniform_contributions,
            mean_deviation_feature_set,
        )
        if (
            edited_average_prediction_score is None
            or constrained_mean_deviation_edited_average is None
        ):
            raise ValueError("Unable to calculate edited average on the full-data reference")
        edited_average_prediction_score += intercept_adjustment
        mean_deviation_constraint = {
            "enabled": should_limit_edited_mean_deviation,
            "applied": bool(scale_factor < (1.0 - 1e-12)),
            "scale_factor": scale_factor,
            "limit": mean_deviation_limit,
            "original_average": mean_deviation_original_average,
            "unconstrained_edited_average": unconstrained_mean_deviation_edited_average,
            "constrained_edited_average": constrained_mean_deviation_edited_average,
            "satisfied": bool(
                abs(
                    constrained_mean_deviation_edited_average
                    - mean_deviation_original_average
                )
                <= mean_deviation_limit + 1e-9
            ),
            "feature_scale_factors": {
                feature_name: scale_factor
                for feature_name in explicitly_scaled_mean_deviation_feature_set
            },
        }
        original_test_set_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=left_artifact,
                evaluated_offsets={},
                method="test_set",
            )
        )
        edited_test_set_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets=right_offsets,
                feature_shifts=right_effective_feature_shifts,
                method="test_set",
            )
        )
        reduced_test_set_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets={},
                method="test_set",
            )
        )
        original_test_set_average_prediction_score = score_from_contributions(
            left_artifact,
            original_test_set_contributions,
        )
        reduced_test_set_average_prediction_score = score_from_contributions(
            right_artifact,
            reduced_test_set_contributions,
        )
        edited_test_set_average_prediction_score = score_from_contributions(
            right_artifact,
            edited_test_set_contributions,
        )
        if edited_test_set_average_prediction_score is not None:
            edited_test_set_average_prediction_score += intercept_adjustment

        missing_indicator_features = self._compare_missing_indicator_features(
            left_artifact, right_artifact
        )
        average_feature_names = [
            feature_name
            for feature_name in dict.fromkeys(
                (left_artifact.get("feature_names") or [])
                + (right_artifact.get("feature_names") or [])
            )
            if feature_name not in missing_indicator_features
        ]
        # The original artifact's exported 80% training partition is the
        # development reference, regardless of the currently active page model.
        main_dataset_rows = left_artifact.get("training_feature_rows") or []
        right_main_dataset_rows = (
            self._project_compare_rows(
                main_dataset_rows, left_artifact, right_artifact,
            ) if main_dataset_rows else []
        )
        original_main_dataset_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=left_artifact,
                evaluated_offsets={},
                method="main_dataset",
                sample_rows=main_dataset_rows,
            )
        )
        edited_main_dataset_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets=right_offsets,
                feature_shifts=right_effective_feature_shifts,
                method="main_dataset",
                sample_rows=right_main_dataset_rows,
            )
        )
        reduced_main_dataset_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets={},
                method="main_dataset",
                sample_rows=right_main_dataset_rows,
            )
        )
        original_main_dataset_average_prediction_score = score_from_contributions(
            left_artifact, original_main_dataset_contributions
        )
        reduced_main_dataset_average_prediction_score = score_from_contributions(
            right_artifact, reduced_main_dataset_contributions
        )
        edited_main_dataset_average_prediction_score = score_from_contributions(
            right_artifact, edited_main_dataset_contributions
        )
        main_dataset_average_prediction_shift = (
            edited_main_dataset_average_prediction_score
            - reduced_main_dataset_average_prediction_score
            if edited_main_dataset_average_prediction_score is not None
            and reduced_main_dataset_average_prediction_score is not None
            else None
        )
        if edited_main_dataset_average_prediction_score is not None:
            edited_main_dataset_average_prediction_score += intercept_adjustment

        def build_contribution_rows(
            original_contributions: Optional[Dict[str, float]],
            reduced_contributions: Optional[Dict[str, float]],
            edited_contributions: Optional[Dict[str, float]],
            feature_names: Optional[List[str]] = None,
        ) -> Optional[List[Dict[str, Any]]]:
            if (
                original_contributions is None
                or reduced_contributions is None
                or edited_contributions is None
            ):
                return None

            original_base = float(left_artifact.get("base_prediction_offset", 0.0) or 0.0)
            reduced_base = original_edited_base
            edited_base = adjusted_edited_base
            rows: List[Dict[str, Any]] = [
                {
                    "feature_name": "__base_prediction_offset__",
                    "original": original_base,
                    "reduced": reduced_base,
                    "edited": edited_base,
                    "difference": edited_base - original_base,
                    "is_base_offset": True,
                }
            ]
            public_features = set(left_artifact.get("selected_feature_columns") or [])
            for feature_name in (
                feature_names or left_artifact.get("selected_feature_columns") or []
            ):
                if feature_name in missing_indicator_features:
                    continue
                if feature_names is None and (
                    feature_name not in original_contributions
                    or feature_name not in edited_contributions
                ):
                    continue
                original_value = float(original_contributions.get(feature_name, 0.0))
                reduced_value = float(reduced_contributions.get(feature_name, 0.0))
                edited_value = float(edited_contributions.get(feature_name, 0.0))
                rows.append(
                    {
                        "feature_name": str(feature_name),
                        "original": original_value,
                        "reduced": reduced_value,
                        "edited": edited_value,
                        "difference": edited_value - original_value,
                        "is_base_offset": False,
                        "is_internal_feature": feature_name not in public_features,
                    }
                )
            return rows

        feature_previews = []
        overall_shape_function_mae_metrics = []
        expert_complete_model_mae_metrics = []
        for feature_name in active_features:
            left_shape = _json_safe(left_shape_map.get(feature_name) or {})
            right_shape = _json_safe(right_shape_map.get(feature_name) or {})
            if not left_shape or not right_shape:
                continue

            left_x_values = list(left_shape.get("x_values") or [])
            right_x_values = list(right_shape.get("x_values") or [])
            left_base_y_values = [float(value) for value in (left_shape.get("y_values") or [])]
            right_model_base_y_values = [
                float(value) for value in (right_shape.get("y_values") or [])
            ]
            if len(left_x_values) != len(left_base_y_values) or len(right_x_values) != len(right_model_base_y_values):
                continue

            right_base_y_values = list(right_model_base_y_values)

            left_effective_y_values = []
            for x_value, base_y in zip(left_x_values, left_base_y_values):
                offset = self._get_compare_offset_for_feature_value(
                    shape_function=left_shape,
                    original_is_categorical=feature_name in left_cat_features,
                    value=x_value,
                    offsets=left_offsets.get(feature_name, {}),
                )
                left_effective_y_values.append(float(base_y) + float(offset))

            right_effective_y_values = []
            right_alternate_confidence_y_values = []
            for x_value, base_y in zip(right_x_values, right_model_base_y_values):
                offset = self._get_compare_offset_for_feature_value(
                    shape_function=right_shape,
                    original_is_categorical=feature_name in right_cat_features,
                    value=x_value,
                    offsets=right_offsets.get(feature_name, {}),
                )
                right_effective_y_values.append(
                    float(base_y)
                    + float(offset)
                    + float(right_effective_feature_shifts.get(feature_name, 0.0))
                )
                alternate_offset = self._get_compare_offset_for_feature_value(
                    shape_function=right_shape,
                    original_is_categorical=feature_name in right_cat_features,
                    value=x_value,
                    offsets=alternate_confidence_offsets.get(feature_name, {}),
                )
                right_alternate_confidence_y_values.append(
                    float(base_y)
                    + float(alternate_offset)
                    + float(alternate_confidence_feature_shifts.get(feature_name, 0.0))
                )

            def build_individual_edit_traces(
                submission_offsets: List[Dict[str, Any]],
                *,
                shape_function: Dict[str, Any],
                is_categorical: bool,
                x_values: List[Any],
                base_y_values: List[float],
                feature_shift: float = 0.0,
            ) -> List[Dict[str, Any]]:
                return [
                    {
                        "submission_id": trace["submission_id"],
                        "user_name": _format_compare_chart_trace_user_name(
                            trace["user_name"]
                        ),
                        "sureness": trace["sureness"],
                        "y_values": [
                            float(base_y)
                            + self._get_compare_offset_for_feature_value(
                                shape_function=shape_function,
                                original_is_categorical=is_categorical,
                                value=x_value,
                                offsets=trace["offsets"],
                            )
                            + float(feature_shift)
                            for x_value, base_y in zip(x_values, base_y_values)
                        ],
                    }
                    for trace in submission_offsets
                ]

            left_individual_edit_traces = build_individual_edit_traces(
                left_submission_offsets.get(feature_name, []),
                shape_function=left_shape,
                is_categorical=feature_name in left_cat_features,
                x_values=left_x_values,
                base_y_values=left_base_y_values,
            )
            right_individual_edit_traces = build_individual_edit_traces(
                right_submission_offsets.get(feature_name, []),
                shape_function=right_shape,
                is_categorical=feature_name in right_cat_features,
                x_values=right_x_values,
                base_y_values=right_model_base_y_values,
                feature_shift=right_effective_feature_shifts.get(feature_name, 0.0),
            )

            feature_type = str(left_shape.get("feature_type", "numeric"))

            card_x_values = None
            card_original_values = None
            if feature_metrics_on_original_test_set:
                card_x_values = [row.get(feature_name) for row in original_test_rows]
                predict_shape = (
                    self._predict_categorical_shape_value
                    if feature_type == "categorical"
                    else self._predict_numeric_shape_value
                )
                card_original_values = [
                    predict_shape(left_shape, value) for value in card_x_values
                ]
                card_baseline_values = [
                    predict_shape(right_shape, value) for value in card_x_values
                ]
                card_edited_values = [
                    baseline_value
                    + self._get_compare_offset_for_feature_value(
                        shape_function=right_shape,
                        original_is_categorical=feature_name in right_cat_features,
                        value=value,
                        offsets=right_offsets.get(feature_name, {}),
                    )
                    + float(right_effective_feature_shifts.get(feature_name, 0.0))
                    for value, baseline_value in zip(
                        card_x_values, card_baseline_values
                    )
                ]
            elif feature_type == "numeric":
                left_numeric_x = np.asarray(left_x_values, dtype=float)
                right_numeric_x = np.asarray(right_x_values, dtype=float)
                overlap_min = max(float(np.min(left_numeric_x)), float(np.min(right_numeric_x)))
                overlap_max = min(float(np.max(left_numeric_x)), float(np.max(right_numeric_x)))
                if overlap_min > overlap_max:
                    raise ValueError(
                        f"Cannot compute per-feature metrics without overlapping numeric domain for '{feature_name}'"
                    )
                card_x_values = np.linspace(
                    overlap_min,
                    overlap_max,
                    MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS,
                ).tolist()
                card_original_values = self._interpolate_compare_values(
                    left_x_values, left_base_y_values, card_x_values
                )
                card_baseline_values = self._interpolate_compare_values(
                    right_x_values, right_model_base_y_values, card_x_values
                )
                card_edited_values = self._interpolate_compare_values(
                    right_x_values, right_effective_y_values, card_x_values
                )

            def calculate_original_base_mae(
                trace_x_values: List[Any],
                trace_y_values: List[float],
                *,
                trace_shape: Dict[str, Any],
                trace_offsets: Dict[Any, float],
                trace_is_categorical: bool,
                trace_feature_shift: float = 0.0,
            ) -> float:
                if feature_metrics_on_original_test_set:
                    sampled_trace_values = [
                        predict_shape(trace_shape, value)
                        + self._get_compare_offset_for_feature_value(
                            shape_function=trace_shape,
                            original_is_categorical=trace_is_categorical,
                            value=value,
                            offsets=trace_offsets,
                        )
                        + trace_feature_shift
                        for value in card_x_values
                    ]
                    return self._compute_compare_mae(
                        card_original_values, sampled_trace_values
                    )
                if feature_type == "numeric":
                    sampled_trace_values = self._interpolate_compare_values(
                        trace_x_values, trace_y_values, card_x_values
                    )
                    return self._compute_compare_mae(
                        card_original_values, sampled_trace_values
                    )
                return self._compute_compare_mae(left_base_y_values, trace_y_values)

            for trace, submission in zip(
                left_individual_edit_traces,
                left_submission_offsets.get(feature_name, []),
            ):
                trace["original_base_mae"] = calculate_original_base_mae(
                    left_x_values,
                    trace["y_values"],
                    trace_shape=left_shape,
                    trace_offsets=submission["offsets"],
                    trace_is_categorical=feature_name in left_cat_features,
                )
            alternate_submissions = {
                submission["submission_id"]: submission
                for submission in right_alternate_submission_offsets.get(feature_name, [])
            }
            for trace, submission in zip(
                right_individual_edit_traces,
                right_submission_offsets.get(feature_name, []),
            ):
                trace["original_base_mae"] = calculate_original_base_mae(
                    right_x_values,
                    trace["y_values"],
                    trace_shape=right_shape,
                    trace_offsets=submission["offsets"],
                    trace_is_categorical=feature_name in right_cat_features,
                    trace_feature_shift=float(
                        right_effective_feature_shifts.get(feature_name, 0.0)
                    ),
                )
                alternate_submission = alternate_submissions.get(trace["submission_id"])
                if alternate_submission is None:
                    continue
                alternate_y_values = [
                    float(base_y)
                    + self._get_compare_offset_for_feature_value(
                        shape_function=right_shape,
                        original_is_categorical=feature_name in right_cat_features,
                        value=x_value,
                        offsets=alternate_submission["offsets"],
                    )
                    + float(alternate_confidence_feature_shifts.get(feature_name, 0.0))
                    for x_value, base_y in zip(right_x_values, right_model_base_y_values)
                ]
                alternate_mae = calculate_original_base_mae(
                    right_x_values,
                    alternate_y_values,
                    trace_shape=right_shape,
                    trace_offsets=alternate_submission["offsets"],
                    trace_is_categorical=feature_name in right_cat_features,
                    trace_feature_shift=float(
                        alternate_confidence_feature_shifts.get(feature_name, 0.0)
                    ),
                )
                trace["original_base_mae_with_confidence"] = (
                    trace["original_base_mae"] if use_confidence else alternate_mae
                )
                trace["original_base_mae_without_confidence"] = (
                    alternate_mae if use_confidence else trace["original_base_mae"]
                )

            baseline_edited_vs_edited_base_mae = self._compute_compare_mae(
                right_model_base_y_values,
                right_model_base_y_values,
            )
            baseline_edited_vs_edited_base_rmse = self._compute_compare_rmse(
                right_model_base_y_values,
                right_model_base_y_values,
            )
            baseline_edited_vs_edited_base_r2 = self._compute_compare_r2(
                right_model_base_y_values,
                right_model_base_y_values,
            )
            edited_vs_edited_base_mae = self._compute_compare_mae(
                right_model_base_y_values,
                right_effective_y_values,
            )
            edited_vs_edited_base_rmse = self._compute_compare_rmse(
                right_model_base_y_values,
                right_effective_y_values,
            )
            edited_vs_edited_base_r2 = self._compute_compare_r2(
                right_model_base_y_values,
                right_effective_y_values,
            )
            if feature_type == "numeric":
                baseline_edited_vs_original_base_mae = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_model_base_y_values,
                        metric_name="MAE",
                    )
                )
                baseline_edited_vs_original_base_rmse = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_model_base_y_values,
                        metric_name="RMSE",
                    )
                )
                baseline_edited_vs_original_base_r2 = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_model_base_y_values,
                        metric_name="R2",
                    )
                )
                edited_vs_original_base_mae = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_effective_y_values,
                        metric_name="MAE",
                    )
                )
                edited_vs_original_base_rmse = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_effective_y_values,
                        metric_name="RMSE",
                    )
                )
                edited_vs_original_base_r2 = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_effective_y_values,
                        metric_name="R2",
                    )
                )
            else:
                baseline_edited_vs_original_base_mae = self._compute_compare_mae(
                    left_base_y_values,
                    right_model_base_y_values,
                )
                baseline_edited_vs_original_base_rmse = self._compute_compare_rmse(
                    left_base_y_values,
                    right_model_base_y_values,
                )
                baseline_edited_vs_original_base_r2 = self._compute_compare_r2(
                    left_base_y_values,
                    right_model_base_y_values,
                )
                edited_vs_original_base_mae = self._compute_compare_mae(
                    left_base_y_values,
                    right_effective_y_values,
                )
                edited_vs_original_base_rmse = self._compute_compare_rmse(
                    left_base_y_values,
                    right_effective_y_values,
                )
                edited_vs_original_base_r2 = self._compute_compare_r2(
                    left_base_y_values,
                    right_effective_y_values,
                )

            if feature_type == "numeric":
                right_numeric_x_values = np.asarray(right_x_values, dtype=float)
                left_numeric_x_values = np.asarray(left_x_values, dtype=float)
                comparison_point_count = int(
                    np.count_nonzero(
                        (left_numeric_x_values >= (np.min(right_numeric_x_values) - 1e-9))
                        & (left_numeric_x_values <= (np.max(right_numeric_x_values) + 1e-9))
                    )
                )
            else:
                comparison_point_count = len(left_base_y_values)
            if feature_name in overall_mae_feature_set:
                overall_shape_function_mae_metrics.append(
                    {
                        "point_count": comparison_point_count,
                        "baseline": baseline_edited_vs_original_base_mae,
                        "edited": edited_vs_original_base_mae,
                    }
                )
            for expert in right_expert_offsets.values():
                expert_y_values = [
                    float(base_y)
                    + self._get_compare_offset_for_feature_value(
                        shape_function=right_shape,
                        original_is_categorical=feature_name in right_cat_features,
                        value=x_value,
                        offsets=expert["offsets"].get(feature_name, {}),
                    )
                    for x_value, base_y in zip(
                        right_x_values,
                        right_model_base_y_values,
                    )
                ]
                expert_mae = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=expert_y_values,
                        metric_name="MAE",
                    )
                    if feature_type == "numeric"
                    else self._compute_compare_mae(
                        left_base_y_values,
                        expert_y_values,
                    )
                )
                expert_complete_model_mae_metrics.append(
                    {
                            "user_name": expert["user_name"],
                            "mae": expert_mae,
                            "baseline_mae": baseline_edited_vs_original_base_mae,
                            "point_count": comparison_point_count,
                        "submission_ids": expert["submission_ids"],
                    }
                )

            card_metrics = {}
            if card_original_values is not None:
                for metric_name, compute_metric, compare_status in (
                    ("mae", self._compute_compare_mae, self._compare_mae_status),
                    ("rmse", self._compute_compare_rmse, self._compare_mae_status),
                    ("r2", self._compute_compare_r2, self._compare_r2_status),
                ):
                    baseline_metric = compute_metric(card_original_values, card_baseline_values)
                    edited_metric = compute_metric(card_original_values, card_edited_values)
                    status_key = (
                        "edited_vs_original_base_status"
                        if metric_name == "mae"
                        else f"edited_vs_original_base_{metric_name}_status"
                    )
                    card_metrics.update(
                        {
                            f"baseline_edited_vs_original_base_{metric_name}": baseline_metric,
                            f"edited_vs_original_base_{metric_name}": edited_metric,
                            status_key: compare_status(edited_metric, baseline_metric),
                        }
                    )

            feature_previews.append(
                {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "left_x_values": left_x_values,
                    "left_x_tick_labels": left_shape.get("x_tick_labels"),
                    "right_x_values": right_x_values,
                    "right_x_tick_labels": right_shape.get("x_tick_labels"),
                    "left_base_y_values": left_base_y_values,
                    "left_effective_y_values": left_effective_y_values,
                    "right_base_y_values": right_base_y_values,
                    "right_effective_y_values": right_effective_y_values,
                    "right_alternate_confidence_y_values": right_alternate_confidence_y_values,
                    "left_individual_edit_traces": left_individual_edit_traces,
                    "right_individual_edit_traces": right_individual_edit_traces,
                    "left_chart_config": left_shape.get("chart_config") or {},
                    "right_chart_config": right_shape.get("chart_config") or {},
                    "comparison_metrics": {
                        "edited_vs_edited_base_mae": edited_vs_edited_base_mae,
                        "edited_vs_original_base_mae": edited_vs_original_base_mae,
                        "baseline_edited_vs_edited_base_mae": baseline_edited_vs_edited_base_mae,
                        "baseline_edited_vs_original_base_mae": baseline_edited_vs_original_base_mae,
                        "edited_vs_edited_base_status": self._compare_mae_status(
                            edited_vs_edited_base_mae,
                            baseline_edited_vs_edited_base_mae,
                        ),
                        "edited_vs_original_base_status": self._compare_mae_status(
                            edited_vs_original_base_mae,
                            baseline_edited_vs_original_base_mae,
                        ),
                        "edited_vs_edited_base_rmse": edited_vs_edited_base_rmse,
                        "edited_vs_original_base_rmse": edited_vs_original_base_rmse,
                        "baseline_edited_vs_edited_base_rmse": baseline_edited_vs_edited_base_rmse,
                        "baseline_edited_vs_original_base_rmse": baseline_edited_vs_original_base_rmse,
                        "edited_vs_edited_base_rmse_status": self._compare_mae_status(
                            edited_vs_edited_base_rmse,
                            baseline_edited_vs_edited_base_rmse,
                        ),
                        "edited_vs_original_base_rmse_status": self._compare_mae_status(
                            edited_vs_original_base_rmse,
                            baseline_edited_vs_original_base_rmse,
                        ),
                        "edited_vs_edited_base_r2": edited_vs_edited_base_r2,
                        "edited_vs_original_base_r2": edited_vs_original_base_r2,
                        "baseline_edited_vs_edited_base_r2": baseline_edited_vs_edited_base_r2,
                        "baseline_edited_vs_original_base_r2": baseline_edited_vs_original_base_r2,
                        "edited_vs_edited_base_r2_status": self._compare_r2_status(
                            edited_vs_edited_base_r2,
                            baseline_edited_vs_edited_base_r2,
                        ),
                        "edited_vs_original_base_r2_status": self._compare_r2_status(
                            edited_vs_original_base_r2,
                            baseline_edited_vs_original_base_r2,
                        ),
                        **card_metrics,
                    },
                }
            )

        main_training_performance_inputs = self._main_training_performance_inputs(
            left_artifact, right_artifact
        )
        if held_out_test_source == "main_training":
            if main_training_performance_inputs is not None:
                training_feature_names, training_targets = (
                    main_training_performance_inputs
                )
                main_training_performance_data = {
                    "feature_rows": _json_safe(
                        self.X_train.loc[:, training_feature_names].to_dict(
                            orient="records"
                        )
                    ),
                    "target_values": [float(value) for value in training_targets],
                }
            else:
                main_training_performance_data = None
            performance_data_artifact = {
                "held_out_test_data": main_training_performance_data,
                "missing_indicator_map": left_artifact.get("missing_indicator_map") or {},
            }
        else:
            performance_data_artifact = (
                left_artifact if held_out_test_source == "left" else right_artifact
            )
        held_out_test_performance = self._build_held_out_test_performance(
            original_artifact=left_artifact,
            edited_artifact=right_artifact,
            test_data_artifact=performance_data_artifact,
            test_data_source=held_out_test_source,
            edited_offsets=right_offsets,
            edited_feature_shifts=right_effective_feature_shifts,
            intercept_adjustment=intercept_adjustment,
            intercept_correction_source=(
                "reduced training rows" if apply_training_intercept_correction
                else "manual"
            ),
        )

        training_mean_prediction_summary = None
        training_rows = right_artifact.get("training_feature_rows") or []
        if training_rows:
            reduced_model_mean = float(np.mean(
                self._predict_model_compare_artifact_rows(
                    right_artifact,
                    training_rows,
                    exclude_missing_indicators=True,
                )
            ))
            edited_before_correction_mean = float(np.mean(
                self._predict_model_compare_artifact_rows(
                    right_artifact,
                    training_rows,
                    offsets=right_offsets,
                    feature_shifts=right_effective_feature_shifts,
                    exclude_missing_indicators=True,
                )
            ))
            difference_before_correction = (
                edited_before_correction_mean - reduced_model_mean
            )
            suggested_matching_adjustment = -difference_before_correction
            training_mean_prediction_summary = {
                "training_row_count": len(training_rows),
                "reduced_model_mean": reduced_model_mean,
                "edited_before_correction_mean": edited_before_correction_mean,
                "difference_before_correction": difference_before_correction,
                "edited_after_correction_mean": (
                    edited_before_correction_mean + intercept_adjustment
                ),
                "original_full_data_intercept": float(
                    left_artifact.get("base_prediction_offset", 0.0) or 0.0
                ),
                "reduced_intercept": original_edited_base,
                "edited_intercept_before_correction": original_edited_base,
                "intercept_adjustment": intercept_adjustment,
                "edited_intercept_after_correction": adjusted_edited_base,
                "suggested_matching_adjustment": suggested_matching_adjustment,
                "suggested_matching_intercept": (
                    original_edited_base + suggested_matching_adjustment
                ),
            }

        return {
            "use_confidence": bool(use_confidence),
            "intercept_adjustment": intercept_adjustment,
            "original_edited_base_prediction_offset": original_edited_base,
            "adjusted_edited_base_prediction_offset": adjusted_edited_base,
            "feature_previews": feature_previews,
            "all_feature_mae_change": self._build_all_feature_mae_change(
                left_artifact=left_artifact,
                right_artifact=right_artifact,
                shared_features=shared_features,
            ),
            "overall_shape_function_mae": {
                **self._build_overall_shape_function_mae(
                    overall_shape_function_mae_metrics,
                ),
                "expert_complete_model_mae": self._build_expert_complete_model_mae(
                    expert_complete_model_mae_metrics,
                ),
            },
            "held_out_test_performance": held_out_test_performance,
            "main_training_performance_available": bool(
                main_training_performance_inputs
            ),
            "average_prediction_scores": {
                "original": original_average_prediction_score,
                "reduced": reduced_average_prediction_score,
                "edited": edited_average_prediction_score,
                "difference": (
                    edited_average_prediction_score - original_average_prediction_score
                ),
                "edited_minus_reduced": (
                    edited_average_prediction_score - reduced_average_prediction_score
                ),
            },
            "training_mean_prediction_summary": training_mean_prediction_summary,
            "test_set_average_prediction_scores": (
                {
                    "original": original_test_set_average_prediction_score,
                    "reduced": reduced_test_set_average_prediction_score,
                    "edited": edited_test_set_average_prediction_score,
                    "difference": (
                        edited_test_set_average_prediction_score
                        - original_test_set_average_prediction_score
                    ),
                    "edited_minus_reduced": (
                        edited_test_set_average_prediction_score
                        - reduced_test_set_average_prediction_score
                    ),
                }
                if original_test_set_average_prediction_score is not None
                and reduced_test_set_average_prediction_score is not None
                and edited_test_set_average_prediction_score is not None
                else None
            ),
            "main_dataset_average_prediction_scores": (
                {
                    "original": original_main_dataset_average_prediction_score,
                    "reduced": reduced_main_dataset_average_prediction_score,
                    "edited": edited_main_dataset_average_prediction_score,
                    "difference": (
                        edited_main_dataset_average_prediction_score
                        - original_main_dataset_average_prediction_score
                    ),
                    "edited_minus_reduced": (
                        edited_main_dataset_average_prediction_score
                        - reduced_main_dataset_average_prediction_score
                    ),
                }
                if original_main_dataset_average_prediction_score is not None
                and reduced_main_dataset_average_prediction_score is not None
                and edited_main_dataset_average_prediction_score is not None
                else None
            ),
            "main_dataset_training_row_count": len(main_dataset_rows),
            "main_dataset_average_prediction_shift": (
                main_dataset_average_prediction_shift
            ),
            "average_prediction_feature_contributions": {
                "uniform": build_contribution_rows(
                    original_uniform_contributions,
                    right_base_uniform_contributions,
                    edited_uniform_contributions,
                ),
                "test_set": build_contribution_rows(
                    original_test_set_contributions,
                    reduced_test_set_contributions,
                    edited_test_set_contributions,
                    average_feature_names,
                ),
                "main_dataset": build_contribution_rows(
                    original_main_dataset_contributions,
                    reduced_main_dataset_contributions,
                    edited_main_dataset_contributions,
                    average_feature_names,
                ),
            },
            "mean_deviation_constraint": mean_deviation_constraint,
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
            "feature_schema": [
                self._with_feature_display_name(feature)
                for feature in (self.feature_schema or [])
            ],
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
            "show_competence_levels": bool(self.show_competence_levels),
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
            "feature_schema": [
                self._with_feature_display_name(feature)
                for feature in self.feature_schema
            ],
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
        baseline_prediction = self._predict_shape_function_sample(baseline)
        return {
            "label": MISSING_CATEGORY_VALUE,
            "count": missing_count,
            "y_value": prediction - baseline_prediction,
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
        if self.model is None or not hasattr(self.model, "get_shape_functions_as_dict"):
            raise ValueError("Native model shape functions are unavailable")
        native_shape = self.model.get_shape_functions_as_dict().get(feature_name)
        if not native_shape:
            raise ValueError(f"No native shape function is available for '{feature_name}'")

        raw_native_x_values = native_shape.get("x", [])
        raw_native_y_values = native_shape.get("y", [])
        native_x_values = (
            list(raw_native_x_values) if raw_native_x_values is not None else []
        )
        native_y_values = (
            list(raw_native_y_values) if raw_native_y_values is not None else []
        )
        if len(native_x_values) != len(native_y_values) or not native_x_values:
            raise ValueError(f"Native shape function for '{feature_name}' is invalid")
        native_is_categorical = (
            str(native_shape.get("datatype", "")).strip() == "categorical"
        )

        def native_effect_at(value: Any) -> float:
            if native_is_categorical:
                try:
                    return float(native_y_values[[str(x) for x in native_x_values].index(str(value))])
                except (ValueError, IndexError, TypeError):
                    return 0.0
            try:
                native_x_numeric = np.asarray(native_x_values, dtype=float)
                native_y_numeric = np.asarray(native_y_values, dtype=float)
                return float(np.interp(float(value), native_x_numeric, native_y_numeric))
            except (TypeError, ValueError):
                return 0.0

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
                shape_values.append(native_effect_at(raw_value))
                effective_x_values.append(raw_value)

            if not effective_x_values:
                is_categorical = False

            if is_categorical:
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
                    shape_values.append(native_effect_at(raw_value))
                    x_values_numeric.append(float(numeric_x))

                missing_bucket = (
                    self._build_numeric_chart_missing_bucket(
                        feature_name,
                        baseline,
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
            shape_values.append(native_effect_at(x_val))

        missing_bucket = (
            self._build_numeric_chart_missing_bucket(
                feature_name,
                baseline,
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
        return encode_numeric_storage_key(x_value)

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
