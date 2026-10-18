"""Inference and optional refit sensitivity for unedited October IQB exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from igann import IGANN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from backend.app.data_processing import _prepare_missing_aware_features, preprocess_data
from backend.app.ml_service import CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN, ImportedIGANNRuntime
from backend.app.preprocessing_sensitivity import _training_only_features
from backend.scripts.iqb_sample_audit import FULL_ARTIFACT, FULL_FILE, REDUCED_ARTIFACT, REDUCED_FILE


def score(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "mean_prediction": float(np.mean(pred)),
            "prediction_sd": float(np.std(pred, ddof=1))}


def differences(a, b):
    diff = np.abs(np.asarray(a) - np.asarray(b))
    return {"mean_abs_prediction_difference": float(diff.mean()),
            "median_abs_prediction_difference": float(np.median(diff)),
            "p95_abs_prediction_difference": float(np.quantile(diff, .95)),
            "max_abs_prediction_difference": float(diff.max()),
            "changed_prediction_n": int(np.count_nonzero(diff > 1e-9))}


def compare(csv_path, artifact_path, refit):
    artifact = json.loads(artifact_path.read_text())
    frame = pd.read_csv(csv_path)
    frame = frame.rename(columns={source: public for source, public in CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()
                                  if source in frame.columns and public not in frame.columns})
    target = artifact["target_column"]
    selected = artifact["selected_feature_columns"]
    valid = pd.to_numeric(frame[target], errors="coerce").notna()
    x = frame.loc[valid, selected].replace("-", pd.NA).copy()
    y = pd.to_numeric(frame.loc[valid, target])
    train_ix, test_ix = train_test_split(x.index.to_numpy(), test_size=.2, random_state=42)
    assert list(map(str, train_ix)) == artifact["training_sample_ids"]
    assert list(map(str, test_ix)) == artifact["held_out_test_data"]["sample_ids"]
    numeric = artifact["num_features"]
    categorical = [col for col in selected if col not in numeric]
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    old_features, old_cats, _, indicators, fills = _prepare_missing_aware_features(x, categorical, numeric)
    old_processed, _ = preprocess_data(old_features, categorical_features=old_cats)
    assert list(old_processed.columns) == artifact["feature_names"]
    assert indicators == artifact["missing_indicator_map"]
    new_train, new_test, new_fills = _training_only_features(x.loc[train_ix], x.loc[test_ix], numeric, categorical)
    assert list(new_test.columns) == artifact["feature_names"]
    old_test = old_processed.loc[test_ix]
    for row, exported in zip(old_test.to_dict("records"), artifact["held_out_test_data"]["feature_rows"]):
        for col in old_test.columns:
            assert np.isclose(row[col], exported[col], atol=1e-7) if col in numeric else row[col] == exported[col]
    model = ImportedIGANNRuntime(feature_names=artifact["feature_names"],
                                 shape_functions=artifact["shape_functions"],
                                 base_prediction_offset=artifact["base_prediction_offset"],
                                 igann_params=artifact["igann_params"],
                                 effective_boosting_rounds=artifact["effective_boosting_rounds"])
    old_prediction = np.asarray(model.predict(old_test)).ravel()
    inference_prediction = np.asarray(model.predict(new_test)).ravel()
    result = {"dataset_name": csv_path.name, "train_n": len(train_ix), "test_n": len(test_ix),
              "historical": score(y.loc[test_ix], old_prediction),
              "fixed_model_train_only_test_fill": score(y.loc[test_ix], inference_prediction),
              "inference_side_prediction_differences": differences(old_prediction, inference_prediction),
              "changed_fills": {col: {"historical": fills[col], "train_only": new_fills[col]}
                                for col in numeric if x[col].isna().any() and abs(fills[col]-new_fills[col]) > 1e-10}}
    if refit:
        params = dict(artifact["igann_params"])
        params.pop("scale_y", None)
        np.random.seed(params["random_state"])
        torch.manual_seed(params["random_state"])
        old_model = IGANN(**params, scale_y=True)
        old_model.fit(old_processed.loc[train_ix], y.loc[train_ix].to_frame(name=target))
        old_refit_prediction = np.asarray(old_model.predict(old_test)).ravel()
        np.random.seed(params["random_state"])
        torch.manual_seed(params["random_state"])
        clean_model = IGANN(**params, scale_y=True)
        clean_model.fit(new_train, y.loc[train_ix].to_frame(name=target))
        clean_prediction = np.asarray(clean_model.predict(new_test)).ravel()
        result["historical_preprocessing_refit"] = score(y.loc[test_ix], old_refit_prediction)
        result["clean_preprocessing_refit"] = score(y.loc[test_ix], clean_prediction)
        result["clean_vs_historical_refit_differences"] = differences(old_refit_prediction, clean_prediction)
        result["clean_vs_exported_historical_differences"] = differences(old_prediction, clean_prediction)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, csv, artifact in (("full", FULL_FILE, FULL_ARTIFACT), ("reduced", REDUCED_FILE, REDUCED_ARTIFACT)):
        results[name] = compare(a.source_dir/csv, a.artifact_dir/artifact, a.refit)
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
    (a.output_dir/"prediction_sensitivity.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
