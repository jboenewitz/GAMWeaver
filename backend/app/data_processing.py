"""Data processing utilities for generic tabular regression datasets."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Determine the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_FILE = PROJECT_ROOT / "bike.csv"

# Numeric columns with these names are commonly integer-encoded categories.
_CATEGORICAL_NAME_HINTS = {
    "season",
    "mnth",
    "month",
    "weekday",
    "workingday",
    "holiday",
    "weathersit",
    "weather",
}


def _looks_like_integer_categorical(series: pd.Series, column_name: str) -> bool:
    """Return True when a numeric series likely represents categories."""
    non_null = series.dropna()
    if non_null.empty:
        return False

    values = pd.to_numeric(non_null, errors="coerce").dropna()
    if values.empty:
        return False

    # Categorical integer-like values should be very close to whole numbers.
    is_integer_like = np.all(np.isclose(values.to_numpy(dtype=float), np.round(values.to_numpy(dtype=float))))
    if not is_integer_like:
        return False

    unique_count = int(values.nunique(dropna=True))
    unique_ratio = unique_count / max(len(values), 1)
    normalized_name = str(column_name).strip().lower()

    if normalized_name in _CATEGORICAL_NAME_HINTS:
        return True

    # General heuristic for low-cardinality integer columns.
    return unique_count <= 12 or (unique_count <= 24 and unique_ratio <= 0.05)


def _resolve_target_column(df: pd.DataFrame, target_column: Optional[str]) -> str:
    """Resolve target column with a bike-compatible fallback."""
    if target_column and target_column in df.columns:
        return target_column
    if "cnt" in df.columns:
        return "cnt"
    if not len(df.columns):
        raise ValueError("Dataset has no columns")
    return str(df.columns[-1])


def _resolve_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]],
) -> List[str]:
    """Resolve selected feature columns and validate against dataset columns."""
    all_columns = [str(col) for col in df.columns]
    available_features = [col for col in all_columns if col != target_column]

    if feature_columns is None:
        selected = list(available_features)
    else:
        requested = [str(col) for col in feature_columns]
        missing = [col for col in requested if col not in all_columns]
        if missing:
            raise ValueError(f"Selected feature columns not found in dataset: {missing}")

        deduped: List[str] = []
        seen = set()
        for col in requested:
            if col == target_column:
                continue
            if col in seen:
                continue
            seen.add(col)
            deduped.append(col)
        selected = deduped

    if not selected:
        raise ValueError("At least one feature column must be selected")

    return selected


def _infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer categorical and numeric features from semantic column patterns."""
    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for column in X.columns:
        series = X[column]
        if is_numeric_dtype(series) and not _looks_like_integer_categorical(series, column):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return categorical_features, numeric_features


def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw dataset from CSV path."""
    resolved_path = Path(csv_path or DATA_FILE)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {resolved_path}")

    df = pd.read_csv(resolved_path)
    if df.empty:
        raise ValueError("Dataset is empty")
    return df


def get_preprocessor(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
):
    """Create and return a preprocessing pipeline with type-specific imputers."""
    if categorical_features is None or numeric_features is None:
        categorical_features, numeric_features = _infer_feature_types(X)

    transformers = []
    if numeric_features:
        num_transformer = Pipeline([("num_imputer", SimpleImputer(strategy="mean"))])
        transformers.append(("num", num_transformer, numeric_features))

    if categorical_features:
        cat_transformer = Pipeline(
            [("cat_imputer", SimpleImputer(strategy="most_frequent"))]
        )
        transformers.append(("cat", cat_transformer, categorical_features))

    if not transformers:
        raise ValueError("No usable features found after preprocessing setup")

    column_transformer = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return column_transformer, categorical_features, numeric_features


def preprocess_data(
    X: pd.DataFrame,
    preprocessor=None,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """Apply preprocessing to the data."""
    if preprocessor is None:
        preprocessor, categorical_features, _ = get_preprocessor(X)
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    # Keep categorical features as object/string-like for IGANN categorical handling.
    if categorical_features:
        for column in categorical_features:
            if column in X_transformed.columns:
                X_transformed[column] = X_transformed[column].astype(str)

    return X_transformed, preprocessor


def prepare_training_data(
    csv_path: Optional[str] = None,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Load, preprocess and split generic tabular data for training."""
    df = load_dataset(csv_path).replace("-", pd.NA)
    resolved_target = _resolve_target_column(df, target_column)

    y_series = pd.to_numeric(df[resolved_target], errors="coerce")
    valid_target_mask = y_series.notna()
    df = df.loc[valid_target_mask].copy()
    y_series = y_series.loc[valid_target_mask]

    if df.empty:
        raise ValueError("No rows remain after dropping invalid target values")

    selected_features = _resolve_feature_columns(df, resolved_target, feature_columns)
    X = df[selected_features].copy()
    if X.empty or len(X.columns) == 0:
        raise ValueError("Dataset must contain at least one feature column")

    categorical_features, numeric_features = _infer_feature_types(X)

    # Ensure numeric columns are coercible to numeric for downstream modeling.
    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X_processed, preprocessor = preprocess_data(
        X,
        preprocessor=None,
        categorical_features=categorical_features,
    )

    y = pd.DataFrame({resolved_target: y_series})

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        df,
        resolved_target,
        selected_features,
        categorical_features,
        numeric_features,
    )
