"""Data processing utilities for persisted dataset workflows."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_FILE = PROJECT_ROOT / "bike.csv"

_BIKE_REQUIRED_COLUMNS = {
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
}

_HOURLY_COLUMN_CANDIDATES = {
    "hr",
    "hour",
    "time of day",
    "time_of_day",
    "time-of-day",
}


def scale_values(values: Any, new_min: float, new_max: float):
    """Scale numeric values to a new range."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        old_min, old_max = values.min(), values.max()
        return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    arr = np.array(values)
    old_min, old_max = arr.min(), arr.max()
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def _decode_csv_bytes(csv_bytes: bytes) -> str:
    """Decode CSV bytes with a few sensible fallbacks."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return csv_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode CSV file")


def read_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    """Read a CSV file from bytes."""
    text = _decode_csv_bytes(csv_bytes)
    return pd.read_csv(StringIO(text))


def read_csv_path(path: str | Path) -> pd.DataFrame:
    """Read a CSV file from disk."""
    return pd.read_csv(Path(path))


def inspect_csv_bytes(csv_bytes: bytes) -> Dict[str, Any]:
    """Inspect raw CSV columns before upload is finalized."""
    df = read_csv_bytes(csv_bytes)
    columns: List[Dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        columns.append(
            {
                "name": str(column),
                "is_numeric": bool(is_numeric_dtype(series)),
                "non_null_count": int(series.notna().sum()),
                "sample_values": [str(v) for v in series.dropna().head(5).tolist()],
            }
        )
    return {
        "columns": columns,
        "row_count": int(len(df)),
    }


def _is_bike_dataset(df: pd.DataFrame, target_column: str) -> bool:
    return target_column == "cnt" and _BIKE_REQUIRED_COLUMNS.issubset(set(df.columns))


def _build_bike_training_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working_df = df.copy()

    working_df["Time of Day"] = working_df["hr"].astype(int)
    working_df["Windspeed"] = scale_values(working_df["windspeed"], 0, 67)
    working_df["Temperature"] = scale_values(working_df["temp"], -8, 39)
    working_df["Perceived Temperature"] = scale_values(working_df["atemp"], -16, 50)
    working_df["Humidity"] = scale_values(working_df["hum"], 0, 100)

    working_df["Season"] = working_df["season"].replace(
        {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    )
    working_df["Weathersituation"] = working_df["weathersit"].replace(
        {
            1: "Clear",
            2: "Cloudy",
            3: "Light Rain",
            4: "Heavy Rain",
        }
    )
    working_df["Type of Day"] = np.where(
        (working_df["workingday"] == 1) & (working_df["holiday"] == 0),
        "Working Day",
        np.where(
            (working_df["workingday"] == 0) & (working_df["holiday"] == 0),
            "Weekend",
            "Holiday",
        ),
    )

    working_df.dropna(subset=["cnt"], inplace=True)
    working_df.replace("-", np.nan, inplace=True)
    working_df.dropna(inplace=True)

    y = pd.DataFrame(working_df["cnt"])
    feature_to_drop = [
        "dteday",
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "cnt",
        "instant",
        "workingday",
        "casual",
        "registered",
        "weekday",
        "Perceived Temperature",
        "Season",
    ]
    X = working_df.drop(columns=feature_to_drop, inplace=False, errors="ignore")
    return X, y, working_df


def _build_generic_training_frame(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working_df = df.copy()
    working_df[target_column] = pd.to_numeric(working_df[target_column], errors="coerce")
    working_df.dropna(subset=[target_column], inplace=True)

    y = pd.DataFrame(working_df[target_column])
    X = working_df.drop(columns=[target_column], inplace=False)
    return X, y, working_df


def _infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = [column for column in X.columns if is_numeric_dtype(X[column])]
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def _guess_numeric_step(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 1.0

    rounded = numeric.round()
    if np.allclose(numeric, rounded):
        return 1.0

    unique_values = np.sort(numeric.unique())
    if len(unique_values) < 2:
        return 0.1

    diffs = np.diff(unique_values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.1

    step = float(np.median(diffs))
    return max(round(step, 6), 0.000001)


def _build_prediction_fields(
    X: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)

    for feature_name in X.columns:
        if feature_name in numeric_set:
            numeric = pd.to_numeric(X[feature_name], errors="coerce").dropna()
            min_value = float(numeric.min()) if not numeric.empty else 0.0
            max_value = float(numeric.max()) if not numeric.empty else 0.0
            default_value = float(numeric.mean()) if not numeric.empty else 0.0
            is_integer = bool(np.allclose(numeric, numeric.round())) if not numeric.empty else False
            fields.append(
                {
                    "name": feature_name,
                    "label": feature_name,
                    "feature_type": "numeric",
                    "default": round(default_value, 6),
                    "min": round(min_value, 6),
                    "max": round(max_value, 6),
                    "step": _guess_numeric_step(numeric),
                    "is_integer": is_integer,
                }
            )
        elif feature_name in categorical_set:
            string_values = X[feature_name].dropna().astype(str)
            unique_values = sorted(string_values.unique().tolist())
            default_value = string_values.mode().iloc[0] if not string_values.empty else ""
            fields.append(
                {
                    "name": feature_name,
                    "label": feature_name,
                    "feature_type": "categorical",
                    "default": str(default_value),
                    "options": unique_values,
                }
            )

    return fields


def _detect_hourly_pattern_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        normalized = str(column).strip().lower()
        if normalized not in _HOURLY_COLUMN_CANDIDATES:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric.empty:
            continue

        rounded = numeric.round()
        if np.allclose(numeric, rounded) and numeric.min() >= 0 and numeric.max() <= 23:
            return str(column)
    return None


def build_dataset_schema(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Infer the persisted schema used for training and prediction."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset")

    target_series = pd.to_numeric(df[target_column], errors="coerce")
    if target_series.dropna().empty:
        raise ValueError(f"Target column '{target_column}' must contain numeric values")

    if _is_bike_dataset(df, target_column):
        dataset_type = "bike_v1"
        X, _, _ = _build_bike_training_frame(df)
    else:
        dataset_type = "generic_csv_v1"
        X, _, _ = _build_generic_training_frame(df, target_column)

    numeric_features, categorical_features = _infer_feature_types(X)
    if not X.columns.tolist():
        raise ValueError("The dataset must contain at least one feature column")

    return {
        "dataset_type": dataset_type,
        "target_column": target_column,
        "target_label": str(target_column),
        "feature_names": [str(column) for column in X.columns.tolist()],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "prediction_fields": _build_prediction_fields(X, numeric_features, categorical_features),
        "capabilities": {
            "hourly_pattern": bool(_detect_hourly_pattern_column(df)),
        },
        "hourly_pattern_column": _detect_hourly_pattern_column(df),
    }


def build_training_frame_from_schema(
    df: pd.DataFrame,
    dataset_schema: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the training frame from a persisted dataset schema."""
    target_column = dataset_schema["target_column"]
    if dataset_schema.get("dataset_type") == "bike_v1":
        X, y, working_df = _build_bike_training_frame(df)
    else:
        X, y, working_df = _build_generic_training_frame(df, target_column)

    feature_names = dataset_schema["feature_names"]
    missing_features = [name for name in feature_names if name not in X.columns]
    if missing_features:
        raise ValueError(
            "Dataset no longer matches its stored schema. Missing features: "
            + ", ".join(missing_features)
        )

    X = X[feature_names].copy()
    return X, y, working_df


def preprocess_features(
    X: pd.DataFrame,
    dataset_schema: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Apply a lightweight preprocessing pass with deterministic fill values."""
    numeric_features = dataset_schema["numeric_features"]
    categorical_features = dataset_schema["categorical_features"]

    transformed = X.copy()
    fill_values: Dict[str, Dict[str, Any]] = {"numeric": {}, "categorical": {}}

    for feature_name in numeric_features:
        transformed[feature_name] = pd.to_numeric(transformed[feature_name], errors="coerce")
        fill_value = float(transformed[feature_name].mean()) if transformed[feature_name].notna().any() else 0.0
        transformed[feature_name] = transformed[feature_name].fillna(fill_value).astype(float)
        fill_values["numeric"][feature_name] = fill_value

    for feature_name in categorical_features:
        series = transformed[feature_name].astype("string")
        non_null = series.dropna().astype(str)
        fill_value = str(non_null.mode().iloc[0]) if not non_null.empty else ""
        transformed[feature_name] = series.fillna(fill_value).astype(str)
        fill_values["categorical"][feature_name] = fill_value

    if categorical_features:
        transformed = transformed.astype({feature_name: "object" for feature_name in categorical_features})

    transformed = transformed[dataset_schema["feature_names"]]
    return transformed, fill_values


def prepare_training_data(
    df: pd.DataFrame,
    dataset_schema: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Prepare processed training and test splits for IGANN training."""
    X, y, working_df = build_training_frame_from_schema(df, dataset_schema)
    X_processed, fill_values = preprocess_features(X, dataset_schema)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, fill_values, working_df


def build_data_summary(df: pd.DataFrame, dataset_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build generic summary statistics for the active dataset."""
    target_column = dataset_schema["target_column"]
    target_series = pd.to_numeric(df[target_column], errors="coerce").dropna()

    return {
        "total_records": int(len(df)),
        "features": dataset_schema["feature_names"],
        "numeric_features": dataset_schema["numeric_features"],
        "categorical_features": dataset_schema["categorical_features"],
        "target_label": dataset_schema["target_label"],
        "target_column": target_column,
        "target_stats": {
            "mean": float(target_series.mean()) if not target_series.empty else 0.0,
            "std": float(target_series.std()) if not target_series.empty else 0.0,
            "min": float(target_series.min()) if not target_series.empty else 0.0,
            "max": float(target_series.max()) if not target_series.empty else 0.0,
        },
    }


def build_hourly_pattern(df: pd.DataFrame, dataset_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build an hourly pattern summary when the dataset supports it."""
    hourly_column = dataset_schema.get("hourly_pattern_column")
    if not dataset_schema.get("capabilities", {}).get("hourly_pattern") or not hourly_column:
        raise ValueError("Hourly pattern is not supported for this dataset")

    target_column = dataset_schema["target_column"]
    working_df = pd.DataFrame(
        {
            "hour": pd.to_numeric(df[hourly_column], errors="coerce"),
            "target": pd.to_numeric(df[target_column], errors="coerce"),
        }
    ).dropna()

    grouped = working_df.groupby("hour")["target"].mean().sort_index()
    return {
        "hours": [int(value) for value in grouped.index.tolist()],
        "avg_target": [float(value) for value in grouped.values.tolist()],
        "target_label": dataset_schema["target_label"],
    }
