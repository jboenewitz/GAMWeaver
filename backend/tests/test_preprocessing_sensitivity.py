import sys
import asyncio
from pathlib import Path
from tempfile import SpooledTemporaryFile

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from sklearn.model_selection import train_test_split
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import preprocessing_sensitivity as sensitivity
from backend.app.data_processing import (
    _prepare_missing_aware_features,
    preprocess_data,
)


def _example():
    frame = pd.DataFrame({
        "target": [float(index * 2 + 1) for index in range(20)],
        "score": [index + 0.25 for index in range(20)],
        "group": ["A" if index % 2 else "B" for index in range(20)],
    })
    train_ids, test_ids = train_test_split(frame.index.to_numpy(), test_size=0.2, random_state=42)
    frame.loc[train_ids[0], "score"] = np.nan
    frame.loc[test_ids[0], "score"] = np.nan
    frame.loc[train_ids[1], "group"] = None
    frame.loc[test_ids[1], "group"] = None
    X = frame[["score", "group"]]
    old_features, old_cat, _, indicator_map, _ = _prepare_missing_aware_features(
        X, ["group"], ["score"]
    )
    old_processed, _ = preprocess_data(old_features, categorical_features=old_cat)
    training_mean = float(frame.loc[train_ids, "target"].mean())
    artifact = {
        "artifact_version": "1.0",
        "model_type": "IGANN",
        "model_source": "trained",
        "dataset_name": "example.csv",
        "target_column": "target",
        "selected_feature_columns": ["score", "group"],
        "feature_names": list(old_processed.columns),
        "num_features": ["score"],
        "cat_features": ["group", "score [missing]"],
        "missing_indicator_map": indicator_map,
        "base_prediction_offset": training_mean,
        "igann_params": {"task": "regression", "n_estimators": 1},
        "effective_boosting_rounds": 0,
        "shape_functions": [
            {"feature_name": "score", "feature_type": "numeric", "x_values": [0, 20], "y_values": [0, 0]},
            {"feature_name": "score [missing]", "feature_type": "categorical", "x_values": ["(Missing)", "Observed"], "y_values": [0, 0]},
            {"feature_name": "group", "feature_type": "categorical", "x_values": ["(Missing)", "A", "B"], "y_values": [0, 0, 0]},
        ],
        "held_out_test_data": {
            "feature_rows": old_processed.loc[test_ids].to_dict(orient="records"),
            "target_values": frame.loc[test_ids, "target"].tolist(),
        },
    }
    return frame, artifact


def test_retraining_uses_train_only_medians_on_same_test_rows(monkeypatch):
    frame, artifact = _example()
    fitted = []
    predicted = []

    class RecordingIGANN:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            fitted.append(X.copy())

        def predict(self, X):
            predicted.append(X.copy())
            return X["score"].to_numpy(dtype=float)

    monkeypatch.setattr(sensitivity, "IGANN", RecordingIGANN)
    result = sensitivity.compare_one_model(frame.to_csv(index=False).encode(), artifact)

    train_ids, test_ids = train_test_split(frame.index.to_numpy(), test_size=0.2, random_state=42)
    full_median = float(frame["score"].median())
    train_median = float(frame.loc[train_ids, "score"].median())
    assert full_median != train_median
    assert fitted[0].loc[train_ids[0], "score"] == full_median
    assert fitted[1].loc[train_ids[0], "score"] == train_median
    assert fitted[1].loc[train_ids[0], "score [missing]"] == "(Missing)"
    assert fitted[1].loc[train_ids[1], "group"] == "(Missing)"
    assert predicted[1].loc[test_ids[0], "score"] == train_median
    assert predicted[1].loc[test_ids[1], "group"] == "(Missing)"
    assert result["numeric_fills"]["score"] == {
        "original": full_median,
        "training_only": train_median,
    }
    assert result["train_rows"] == 16
    assert result["test_rows"] == 4
    assert result["delta"]["mae"] == pytest.approx(
        result["corrected"]["mae"] - result["reproduced_original"]["mae"]
    )


def test_mismatched_original_test_data_is_rejected(monkeypatch):
    frame, artifact = _example()
    artifact["held_out_test_data"]["target_values"][0] += 1

    class NeverTrain:
        def __init__(self, **kwargs):
            raise AssertionError("Training should not start")

    monkeypatch.setattr(sensitivity, "IGANN", NeverTrain)
    with pytest.raises(ValueError, match="test targets do not match"):
        sensitivity.compare_one_model(frame.to_csv(index=False).encode(), artifact)


def test_edited_artifact_is_rejected_before_training():
    frame, artifact = _example()
    artifact["artifact_version"] = "2.0"
    with pytest.raises(ValueError, match="unedited version 1.0"):
        sensitivity.compare_one_model(frame.to_csv(index=False).encode(), artifact)


def test_sensitivity_endpoint_requires_superadmin():
    from backend.app.admin_routes import preprocessing_sensitivity

    file = SpooledTemporaryFile(max_size=1024)
    file.write(b"x")
    file.seek(0)
    upload = UploadFile(file=file, filename="example.csv")
    request = Request({"type": "http", "headers": []})
    with pytest.raises(HTTPException) as error:
        asyncio.run(preprocessing_sensitivity(request, upload, upload, upload, upload))
    assert error.value.status_code == 403
