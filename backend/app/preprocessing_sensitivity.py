"""Compare exported IQB models with training-only missing-value preprocessing."""

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from igann import IGANN
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from .data_processing import (
    MISSING_CATEGORY_VALUE,
    _infer_feature_types,
    _missing_indicator_name,
    _prepare_missing_aware_features,
    preprocess_data,
)
from .ml_service import CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN, ImportedIGANNRuntime


def _metrics(actual: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(root_mean_squared_error(actual, predictions)),
        "r2": float(r2_score(actual, predictions)),
    }


def _training_only_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Learn medians and feature schema on training rows, then transform test rows."""
    fill_values: dict[str, float] = {}
    indicators: list[str] = []
    for name in numeric:
        observed = pd.to_numeric(train[name], errors="coerce").dropna()
        fill_values[name] = float(observed.median()) if not observed.empty else 0.0
        if pd.to_numeric(train[name], errors="coerce").isna().any():
            indicators.append(_missing_indicator_name(name))

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for name in numeric:
            values = pd.to_numeric(frame[name], errors="coerce")
            result[name] = values.fillna(fill_values[name])
            indicator = _missing_indicator_name(name)
            if indicator in indicators:
                result[indicator] = np.where(
                    values.isna(), MISSING_CATEGORY_VALUE, "Observed"
                )
        for name in categorical:
            values = frame[name].astype(object)
            result[name] = values.where(values.notna(), MISSING_CATEGORY_VALUE).astype(str)
        return result

    internal_categorical = categorical + indicators
    processed_train, preprocessor = preprocess_data(
        transform(train), categorical_features=internal_categorical
    )
    processed_test, _ = preprocess_data(
        transform(test), preprocessor=preprocessor,
        categorical_features=internal_categorical,
    )
    return processed_train, processed_test, fill_values


def _verify_exported_test_rows(
    artifact: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series,
    numeric: list[str],
) -> None:
    held_out = artifact.get("held_out_test_data")
    if not held_out:
        return
    rows = held_out.get("feature_rows") or []
    targets = held_out.get("target_values") or []
    if len(rows) != len(X_test) or len(targets) != len(y_test):
        raise ValueError("The CSV split does not match the original model's test set")
    if not np.allclose(np.asarray(targets, dtype=float), y_test.to_numpy(), atol=1e-8, rtol=0):
        raise ValueError("The CSV test targets do not match the original model")
    for actual_row, exported_row in zip(X_test.to_dict(orient="records"), rows):
        for name, value in actual_row.items():
            if name not in exported_row:
                raise ValueError("The original model test data has missing features")
            if name in numeric:
                matches = np.isclose(float(value), float(exported_row[name]), atol=1e-7, rtol=0)
            else:
                matches = str(value) == str(exported_row[name])
            if not matches:
                raise ValueError("The CSV test features do not match the original model")


def compare_one_model(csv_bytes: bytes, artifact: dict[str, Any]) -> dict[str, Any]:
    """Retrain one unedited exported model on the same split with train-only fills."""
    if not isinstance(artifact, dict):
        raise ValueError("The model export must be a JSON object")
    if artifact.get("model_type") != "IGANN" or artifact.get("model_source") != "trained":
        raise ValueError("Upload an unedited, locally trained IGANN model export")
    if artifact.get("artifact_version") != "1.0" or not artifact.get("shape_functions"):
        raise ValueError("Upload an unedited version 1.0 model export")
    if not isinstance(artifact.get("igann_params"), dict):
        raise ValueError("The model export has no IGANN training parameters")
    feature_names = artifact.get("feature_names") or []
    shape_names = [item.get("feature_name") for item in artifact["shape_functions"]]
    if not feature_names or len(shape_names) != len(feature_names) or set(shape_names) != set(feature_names):
        raise ValueError("The model export does not contain every original shape function")
    if not csv_bytes:
        raise ValueError("The CSV is empty")
    try:
        frame = pd.read_csv(BytesIO(csv_bytes)).replace("-", pd.NA)
    except Exception as exc:
        raise ValueError(f"Could not read the CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("The CSV has no data rows")

    target = artifact.get("target_column")
    selected = list(artifact.get("selected_feature_columns") or [])
    # Translate only when an export expects public English headings.
    if any(name not in frame.columns for name in selected):
        frame = frame.rename(columns={
            source: public
            for source, public in CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()
            if source in frame.columns and public not in frame.columns
        })
    if not target or target not in frame.columns or not selected:
        raise ValueError("The CSV does not contain the model target or selected features")
    if any(name not in frame.columns for name in selected):
        raise ValueError("The CSV does not contain all model features")
    y = pd.to_numeric(frame[target], errors="coerce")
    valid = y.notna()
    X = frame.loc[valid, selected].copy()
    y = y.loc[valid]
    if len(X) < 10:
        raise ValueError("Too few rows have a numeric target")

    train_ids, test_ids = train_test_split(X.index.to_numpy(), test_size=0.2, random_state=42)
    categorical, numeric = _infer_feature_types(X)
    training_categorical, training_numeric = _infer_feature_types(X.loc[train_ids])
    exported_numeric = [name for name in artifact.get("num_features", []) if name in selected]
    exported_categorical = [name for name in artifact.get("cat_features", []) if name in selected]
    if numeric != exported_numeric or categorical != exported_categorical:
        raise ValueError("The CSV feature types do not match the original model")
    if training_numeric != numeric or training_categorical != categorical:
        raise ValueError("Feature types differ between the training split and original model")
    for name in numeric:
        X[name] = pd.to_numeric(X[name], errors="coerce")

    old_features, old_categorical, _, old_indicators, old_fills = (
        _prepare_missing_aware_features(X, categorical, numeric)
    )
    old_processed, _ = preprocess_data(
        old_features, categorical_features=old_categorical
    )
    if list(old_processed.columns) != artifact.get("feature_names"):
        raise ValueError("The original model's preprocessed feature schema differs")
    if old_indicators != artifact.get("missing_indicator_map"):
        raise ValueError("The original model's missing-value indicators differ")

    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    if not np.isclose(
        float(artifact.get("base_prediction_offset")), float(y_train.mean()),
        atol=1e-8, rtol=0,
    ):
        raise ValueError("The CSV training split does not match the original model")
    old_test = old_processed.loc[test_ids]
    _verify_exported_test_rows(artifact, old_test, y_test, numeric)

    new_train, new_test, train_fills = _training_only_features(
        X.loc[train_ids], X.loc[test_ids], training_numeric, training_categorical
    )
    if list(new_train.columns) != artifact["feature_names"]:
        raise ValueError("Training-only preprocessing changed the model feature schema")

    original = ImportedIGANNRuntime(
        feature_names=artifact["feature_names"],
        shape_functions=artifact["shape_functions"],
        base_prediction_offset=artifact["base_prediction_offset"],
        igann_params=artifact.get("igann_params") or {},
        effective_boosting_rounds=artifact.get("effective_boosting_rounds") or 0,
    )
    original_metrics = _metrics(y_test, original.predict(old_test))
    params = dict(artifact["igann_params"])
    params.pop("scale_y", None)
    reproduced_original = IGANN(**params, scale_y=True)
    reproduced_original.fit(old_processed.loc[train_ids], y_train.to_frame(name=target))
    reproduced_metrics = _metrics(y_test, reproduced_original.predict(old_test))
    corrected_model = IGANN(**params, scale_y=True)
    corrected_model.fit(new_train, y_train.to_frame(name=target))
    corrected_metrics = _metrics(y_test, corrected_model.predict(new_test))

    return {
        "dataset_name": artifact.get("dataset_name"),
        "source_rows": int(len(frame)),
        "usable_rows": int(len(X)),
        "train_rows": int(len(train_ids)),
        "test_rows": int(len(test_ids)),
        "original": original_metrics,
        "reproduced_original": reproduced_metrics,
        "corrected": corrected_metrics,
        "delta": {
            name: corrected_metrics[name] - reproduced_metrics[name]
            for name in reproduced_metrics
        },
        "numeric_fills": {
            name: {"original": old_fills[name], "training_only": train_fills[name]}
            for name in numeric if X[name].isna().any()
        },
        "original_metric_source": "exported_shape_functions",
    }


def compare_models(
    full_csv: bytes, full_artifact: dict[str, Any],
    reduced_csv: bytes, reduced_artifact: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(full_artifact, dict) or not isinstance(reduced_artifact, dict):
        raise ValueError("Both model exports must be JSON objects")
    if full_artifact.get("target_column") != reduced_artifact.get("target_column"):
        raise ValueError("The two models must use the same target")
    normalize = lambda artifact: [
        CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.get(name, name)
        for name in artifact.get("selected_feature_columns") or []
    ]
    if normalize(full_artifact) != normalize(reduced_artifact):
        raise ValueError("The two models must use the same IQB predictors")
    full = compare_one_model(full_csv, full_artifact)
    reduced = compare_one_model(reduced_csv, reduced_artifact)
    if full["usable_rows"] <= reduced["usable_rows"]:
        raise ValueError("The full-data CSV must contain more usable rows than the reduced CSV")
    return {"full": full, "reduced": reduced}
