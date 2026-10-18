"""Historical-preprocessing IGANN refits under alternative 5% source samples."""

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

from backend.app.data_processing import MISSING_CATEGORY_VALUE, _prepare_missing_aware_features, preprocess_data
from backend.app.ml_service import CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN, ImportedIGANNRuntime
from backend.scripts.iqb_sample_audit import FULL_ARTIFACT, FULL_FILE, REDUCED_ARTIFACT


def metric(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred))}


def shape_mae(ref, other):
    if ref["feature_type"] == "categorical":
        observed = dict(zip(map(str, other["x_values"]), map(float, other["y_values"])))
        x, a = list(map(str, ref["x_values"])), np.asarray(ref["y_values"], float)
        keep = [j for j, level in enumerate(x) if level in observed]
        return float(np.mean([abs(a[j]-observed[x[j]]) for j in keep])) if keep else np.nan
    rx, ry = np.asarray(ref["x_values"], float), np.asarray(ref["y_values"], float)
    ox, oy = np.asarray(other["x_values"], float), np.asarray(other["y_values"], float)
    lo, hi = max(rx.min(), ox.min()), min(rx.max(), ox.max())
    if hi < lo:
        return np.nan
    grid = np.linspace(lo, hi, 100)
    return float(np.mean(np.abs(np.interp(grid, rx, ry)-np.interp(grid, ox, oy))))


def native_shapes(model):
    result = {}
    for name, sf in model.get_shape_functions_as_dict().items():
        result[name] = {"feature_name": name, "feature_type": "categorical" if sf.get("datatype") == "categorical" else "numeric",
                        "x_values": np.asarray(sf["x"]).tolist(), "y_values": np.asarray(sf["y"]).tolist()}
    return result


def prepare_sample(full, source_ix, target, selected, numeric, categorical, test_ix):
    frame = full.iloc[source_ix]
    valid = pd.to_numeric(frame[target], errors="coerce").notna()
    frame = frame.loc[valid]
    train_positions, _ = train_test_split(np.arange(len(frame)), test_size=.2, random_state=42)
    train_source_ix = frame.index.to_numpy()[train_positions]
    x = frame[selected].replace("-", pd.NA).copy()
    for name in numeric:
        x[name] = pd.to_numeric(x[name], errors="coerce")
    transformed, cats, _, indicators, fills = _prepare_missing_aware_features(x, categorical, numeric)
    transformed, _ = preprocess_data(transformed, categorical_features=cats)
    expected_indicators = [name+" [missing]" for name in numeric if x[name].isna().any()]
    assert set(indicators.values()) == set(expected_indicators)
    eval_raw = full.loc[test_ix, selected].replace("-", pd.NA)
    eval_processed = pd.DataFrame(index=eval_raw.index)
    for name in numeric:
        v = pd.to_numeric(eval_raw[name], errors="coerce")
        eval_processed[name] = v.fillna(fills[name])
        if name in indicators:
            eval_processed[indicators[name]] = np.where(v.isna(), MISSING_CATEGORY_VALUE, "Observed")
    for name in categorical:
        v = eval_raw[name].astype(object)
        eval_processed[name] = v.where(v.notna(), MISSING_CATEGORY_VALUE).astype(str)
    eval_processed, _ = preprocess_data(eval_processed, categorical_features=cats)
    return transformed.loc[train_source_ix], pd.to_numeric(frame.loc[train_source_ix, target]), eval_processed, train_source_ix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models", type=int, default=50)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fa = json.loads((args.artifact_dir/FULL_ARTIFACT).read_text())
    ra = json.loads((args.artifact_dir/REDUCED_ARTIFACT).read_text())
    raw = pd.read_csv(args.source_dir/FULL_FILE)
    raw = raw.rename(columns={source: public for source, public in CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()
                              if source in raw.columns and public not in raw.columns})
    target, selected, numeric = fa["target_column"], fa["selected_feature_columns"], fa["num_features"]
    categorical = [name for name in selected if name not in numeric]
    actual_source_ix = np.sort(np.random.default_rng(42).choice(len(raw), size=round(.05*len(raw)), replace=False))
    actual_valid_ix = actual_source_ix[pd.to_numeric(raw.iloc[actual_source_ix][target], errors="coerce").notna().to_numpy()]
    actual_train_pos, _ = train_test_split(np.arange(len(actual_valid_ix)), test_size=.2, random_state=42)
    actual_train_source = set(actual_valid_ix[actual_train_pos])
    rng = np.random.default_rng(20261008)
    sampled_ix = [np.sort(rng.choice(len(raw), size=len(actual_source_ix), replace=False)) for _ in range(args.models)]
    train_sets = []
    for ix in sampled_ix:
        valid_ix = ix[pd.to_numeric(raw.iloc[ix][target], errors="coerce").notna().to_numpy()]
        train_pos, _ = train_test_split(np.arange(len(valid_ix)), test_size=.2, random_state=42)
        train_sets.append(set(valid_ix[train_pos]))
    full_test = set(map(int, fa["held_out_test_data"]["sample_ids"]))
    common = sorted(full_test-actual_train_source-set().union(*train_sets))
    if len(common) < 100:
        raise ValueError("Too few evaluation rows remain untouched by all models")
    actual = pd.to_numeric(raw.loc[common, target]).to_numpy(float)
    ref_shapes = {s["feature_name"]: s for s in fa["shape_functions"]}
    actual_shapes = {s["feature_name"]: s for s in ra["shape_functions"]}
    actual_model = ImportedIGANNRuntime(feature_names=ra["feature_names"], shape_functions=ra["shape_functions"],
                                        base_prediction_offset=ra["base_prediction_offset"], igann_params=ra["igann_params"],
                                        effective_boosting_rounds=ra["effective_boosting_rounds"])
    # Predict each October artifact with its own historical fitted fills.
    def artifact_eval(artifact):
        values = pd.DataFrame(index=common)
        fills = {}
        for name, indicator in artifact["missing_indicator_map"].items():
            fills[name] = next(float(r[name]) for r in artifact["training_feature_rows"] if r[indicator] == MISSING_CATEGORY_VALUE)
        for name in numeric:
            v = pd.to_numeric(raw.loc[common, name], errors="coerce")
            values[name] = v.fillna(fills.get(name, float(v.median())))
            if name in artifact["missing_indicator_map"]:
                values[artifact["missing_indicator_map"][name]] = np.where(v.isna(), MISSING_CATEGORY_VALUE, "Observed")
        for name in categorical:
            v = raw.loc[common, name].astype(object)
            values[name] = v.where(v.notna(), MISSING_CATEGORY_VALUE).astype(str)
        return values.loc[:, artifact["feature_names"]]
    actual_pred = np.asarray(actual_model.predict(artifact_eval(ra))).ravel()
    full_model = ImportedIGANNRuntime(feature_names=fa["feature_names"], shape_functions=fa["shape_functions"],
                                      base_prediction_offset=fa["base_prediction_offset"], igann_params=fa["igann_params"],
                                      effective_boosting_rounds=fa["effective_boosting_rounds"])
    full_pred = np.asarray(full_model.predict(artifact_eval(fa))).ravel()
    actual_shape_mae = {name: shape_mae(ref_shapes[name], actual_shapes[name]) for name in selected}
    params = dict(ra["igann_params"])
    params.pop("scale_y", None)
    actual_x, actual_y, actual_eval, _ = prepare_sample(
        raw, actual_source_ix, target, selected, numeric, categorical, common
    )
    np.random.seed(params["random_state"])
    torch.manual_seed(params["random_state"])
    actual_refit = IGANN(**params, scale_y=True)
    actual_refit.fit(actual_x, actual_y.to_frame(name=target))
    actual_refit_shapes = native_shapes(actual_refit)
    actual_refit_pred = np.asarray(actual_refit.predict(actual_eval)).ravel()
    actual_refit_metrics = metric(actual, actual_refit_pred)
    actual_refit_shape_mae = {
        name: shape_mae(ref_shapes[name], actual_refit_shapes[name]) for name in selected
    }
    reproduction = {
        "prediction_mae_export_vs_refit": float(np.mean(np.abs(actual_pred-actual_refit_pred))),
        "shape_mae_export_vs_refit": {
            name: shape_mae(actual_shapes[name], actual_refit_shapes[name]) for name in selected
        },
    }
    metrics, shapes = [], []
    for j, ix in enumerate(sampled_ix, 1):
        x_train, y_train, x_test, train_ix = prepare_sample(raw, ix, target, selected, numeric, categorical, common)
        assert x_train.columns.tolist() == ra["feature_names"]
        np.random.seed(params["random_state"])
        torch.manual_seed(params["random_state"])
        model = IGANN(**params, scale_y=True)
        model.fit(x_train, y_train.to_frame(name=target))
        pred = np.asarray(model.predict(x_test)).ravel()
        metrics.append({"replicate": j, "training_n": len(train_ix), **metric(actual, pred)})
        shape_map = native_shapes(model)
        for name in selected:
            shapes.append({"replicate": j, "feature": name, "shape_mae": shape_mae(ref_shapes[name], shape_map[name])})
        if j % 10 == 0:
            print(f"completed {j}/{args.models}", flush=True)
    metrics = pd.DataFrame(metrics)
    shapes = pd.DataFrame(shapes)
    metrics.to_csv(args.output_dir/"model_instability_metrics.csv", index=False, float_format="%.10g")
    shapes.to_csv(args.output_dir/"model_instability_shapes.csv", index=False, float_format="%.10g")
    actual_metrics = metric(actual, actual_pred)
    rows = []
    for key in ("mae", "rmse", "r2"):
        vals = metrics[key].to_numpy()
        rows.append({"statistic": key, "feature": "", "actual": actual_metrics[key],
                     "actual_native_refit": actual_refit_metrics[key],
                     "actual_native_refit_percentile": 100*np.mean(vals <= actual_refit_metrics[key]),
                     "median": np.median(vals),
                     "q05": np.quantile(vals, .05), "q95": np.quantile(vals, .95),
                     "percentile_rank": 100*np.mean(vals <= actual_metrics[key])})
    for name in selected:
        vals = shapes.loc[shapes.feature == name, "shape_mae"].to_numpy()
        rows.append({"statistic": "shape_mae", "feature": name, "actual": actual_shape_mae[name],
                     "actual_native_refit": actual_refit_shape_mae[name],
                     "actual_native_refit_percentile": 100*np.mean(vals <= actual_refit_shape_mae[name]),
                     "median": np.median(vals), "q05": np.quantile(vals, .05), "q95": np.quantile(vals, .95),
                     "percentile_rank": 100*np.mean(vals <= actual_shape_mae[name])})
    pd.DataFrame(rows).to_csv(args.output_dir/"model_instability_positions.csv", index=False, float_format="%.10g")
    summary = {"models": args.models, "rng_seed": 20261008, "common_evaluation_n": len(common),
               "full_reference_metrics": metric(actual, full_pred), "actual_reduced_metrics": actual_metrics,
               "actual_reduced_native_refit_metrics": actual_refit_metrics,
               "actual_refit_reproduction": reproduction,
               "shape_mae_method": "100 equally spaced points on pairwise overlapping numeric range; category mean absolute curve difference on common levels",
               "preprocessing": "historical pre-split target-valid medians for every alternative fit"}
    (args.output_dir/"model_instability_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
