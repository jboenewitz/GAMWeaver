"""Artifact constants, JSON conversion, and the imported IGANN runtime."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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
