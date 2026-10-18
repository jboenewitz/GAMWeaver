"""Aggregate audit of the historical IQB 5% sample; no model fitting.

Run from the repository root with /usr/bin/python3 -m backend.scripts.iqb_sample_audit.
Only aggregate statistics are written; source rows remain in their original files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from backend.app.ml_service import CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN


PREFIX = (
    "IQB-BT_2021_student_quest_v1_SUF_Off-site_2606-02a_"
    "german_reading_selected15_v8_de_"
)
FULL_FILE = PREFIX + "random100pct_pv_mean_old.csv"
REDUCED_FILE = PREFIX + "random5pct_pv_mean.csv"
FULL_ARTIFACT = "original_model_for_real.json"
REDUCED_ARTIFACT = "igann-model-2026-10-07T10-00-24-717427+00-00.json"
SEED = 42
REPLICATES = 2000


def number(value):
    return None if not np.isfinite(value) else float(value)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")


def target_stats(values: pd.Series) -> dict:
    a = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n": len(a), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
        "median": float(np.median(a)), "q05": float(np.quantile(a, .05)),
        "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75)),
        "q95": float(np.quantile(a, .95)), "minimum": float(a.min()),
        "maximum": float(a.max()),
    }


def empirical_position(draws: np.ndarray, actual: float) -> dict:
    return {
        "actual": float(actual),
        "percentile_less_or_equal": float(100 * np.mean(draws <= actual)),
        "two_sided_tail_percent": float(200 * min(np.mean(draws <= actual), np.mean(draws >= actual))),
        "mc_median": float(np.median(draws)),
        "mc_q025": float(np.quantile(draws, .025)),
        "mc_q975": float(np.quantile(draws, .975)),
        "mc_min": float(draws.min()), "mc_max": float(draws.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("Use at least 100 Monte Carlo replicates")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.source_dir / FULL_FILE)
    reduced = pd.read_csv(args.source_dir / REDUCED_FILE)
    with (args.artifact_dir / FULL_ARTIFACT).open(encoding="utf-8") as handle:
        full_artifact = json.load(handle)
    with (args.artifact_dir / REDUCED_ARTIFACT).open(encoding="utf-8") as handle:
        reduced_artifact = json.load(handle)
    assert full.columns.equals(reduced.columns)
    assert full_artifact["dataset_name"] == FULL_FILE
    assert reduced_artifact["dataset_name"] == REDUCED_FILE
    assert full_artifact["preprocessing_scope"] == reduced_artifact["preprocessing_scope"] == "whole_dataset_before_split"
    assert full_artifact["selected_feature_columns"] == reduced_artifact["selected_feature_columns"]
    actual_indices = np.sort(np.random.default_rng(SEED).choice(
        len(full), size=round(len(full) * .05), replace=False
    ))
    assert full.iloc[actual_indices].reset_index(drop=True).equals(reduced)
    source_by_public = {v: k for k, v in CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()}
    selected = [source_by_public.get(x, x) for x in full_artifact["selected_feature_columns"]]
    numeric = {source_by_public.get(x, x) for x in full_artifact["num_features"]}
    assert set(selected).issubset(full.columns)
    target = full_artifact["target_column"]
    full_valid = full.loc[full[target].notna()]
    reduced_valid = reduced.loc[reduced[target].notna()]
    assert len(full_valid) == 26244 and len(reduced_valid) == 1309
    full_train, full_test = train_test_split(full_valid.index.to_numpy(), test_size=.2, random_state=SEED)
    reduced_train, reduced_test = train_test_split(reduced_valid.index.to_numpy(), test_size=.2, random_state=SEED)
    assert len(full_train) == 20995 and len(full_test) == 5249
    assert len(reduced_train) == 1047 and len(reduced_test) == 262
    assert full_artifact["training_sample_ids"] == [str(x) for x in full_train]
    assert reduced_artifact["training_sample_ids"] == [str(x) for x in reduced_train]
    assert full_artifact["held_out_test_data"]["sample_ids"] == [str(x) for x in full_test]
    assert reduced_artifact["held_out_test_data"]["sample_ids"] == [str(x) for x in reduced_test]
    reduced_train_source = set(actual_indices[reduced_train])
    overlap = reduced_train_source.intersection(full_test)
    assert len(overlap) == 219

    target_rows = []
    for cohort, frame in (("full_raw", full), ("reduced_raw", reduced),
                          ("full_target_valid", full_valid), ("reduced_target_valid", reduced_valid),
                          ("full_training", full_valid.loc[full_train]),
                          ("reduced_training", reduced_valid.loc[reduced_train])):
        row = {"cohort": cohort, "rows": len(frame), "target_missing": int(frame[target].isna().sum())}
        row.update(target_stats(frame[target]))
        target_rows.append(row)
    save_table(pd.DataFrame(target_rows), out / "target_distribution.csv")
    full_target = full_valid[target].to_numpy(dtype=float)
    reduced_target = reduced_valid[target].to_numpy(dtype=float)
    full_target_sd = float(full_target.std(ddof=1))
    target_comparison = {
        "mean_difference": float(reduced_target.mean() - full_target.mean()),
        "standardized_mean_difference": float((reduced_target.mean() - full_target.mean()) / full_target_sd),
        "sd_ratio": float(reduced_target.std(ddof=1) / full_target_sd),
        "ks_statistic": float(ks_2samp(reduced_target, full_target).statistic),
    }

    missing_rows = []
    feature_rows = []
    support_rows = []
    for col in selected:
        for scope, a, b in (("target_valid", full_valid, reduced_valid), ("all_source", full, reduced)):
            fm = int(a[col].isna().sum())
            rm = int(b[col].isna().sum())
            missing_rows.append({
                "scope": scope, "feature": col, "type": "numeric" if col in numeric else "categorical",
                "full_rows": len(a), "reduced_rows": len(b),
                "full_missing_n": fm, "reduced_missing_n": rm,
                "full_missing_pct": 100 * fm / len(a), "reduced_missing_pct": 100 * rm / len(b),
                "difference_pp": 100 * (rm / len(b) - fm / len(a)),
            })
        a = full_valid[col].dropna()
        b = reduced_valid[col].dropna()
        if col in numeric:
            aa = a.to_numpy(dtype=float)
            bb = b.to_numpy(dtype=float)
            sd = float(aa.std(ddof=1))
            feature_rows.append({
                "feature": col, "type": "numeric", "level": "", "full_n": len(aa), "reduced_n": len(bb),
                "full_mean_or_share": float(aa.mean()), "reduced_mean_or_share": float(bb.mean()),
                "difference": float(bb.mean() - aa.mean()), "full_sd": sd,
                "standardized_difference": number((bb.mean() - aa.mean()) / sd) if sd else None,
            })
            support_rows.append({
                "feature": col, "type": "numeric", "level": "observed_values",
                "full_n": len(aa), "reduced_n": len(bb),
                "full_unique": int(a.nunique()), "reduced_unique": int(b.nunique()),
                "full_min": float(aa.min()), "reduced_min": float(bb.min()),
                "full_max": float(aa.max()), "reduced_max": float(bb.max()),
                "full_q05": float(np.quantile(aa, .05)), "reduced_q05": float(np.quantile(bb, .05)),
                "full_q95": float(np.quantile(aa, .95)), "reduced_q95": float(np.quantile(bb, .95)),
            })
        else:
            levels = sorted(set(a.unique()) | set(b.unique()))
            for level in levels:
                p = float((a == level).mean())
                q = float((b == level).mean())
                denom = np.sqrt(p * (1 - p))
                feature_rows.append({
                    "feature": col, "type": "categorical", "level": str(level),
                    "full_n": int((a == level).sum()), "reduced_n": int((b == level).sum()),
                    "full_mean_or_share": p, "reduced_mean_or_share": q,
                    "difference": q - p, "full_sd": denom,
                    "standardized_difference": number((q - p) / denom) if denom else None,
                })
                support_rows.append({
                    "feature": col, "type": "categorical", "level": str(level),
                    "full_n": int((a == level).sum()), "reduced_n": int((b == level).sum()),
                    "full_unique": len(levels), "reduced_unique": int(b.nunique()),
                    "full_min": None, "reduced_min": None, "full_max": None, "reduced_max": None,
                    "full_q05": None, "reduced_q05": None, "full_q95": None, "reduced_q95": None,
                })
    missing = pd.DataFrame(missing_rows)
    features = pd.DataFrame(feature_rows)
    support = pd.DataFrame(support_rows)
    save_table(missing, out / "missingness.csv")
    save_table(features, out / "standardized_differences.csv")
    save_table(support, out / "feature_support.csv")
    level_rows = []
    for col in selected:
        full_counts = full_valid[col].value_counts(dropna=True)
        reduced_counts = reduced_valid[col].value_counts(dropna=True)
        full_training_counts = full_valid.loc[full_train, col].value_counts(dropna=True)
        reduced_training_counts = reduced_valid.loc[reduced_train, col].value_counts(dropna=True)
        for level in sorted(full_counts.index):
            level_rows.append({
                "feature": col, "type": "numeric" if col in numeric else "categorical",
                "level": str(level),
                "full_valid_n": int(full_counts.get(level, 0)),
                "reduced_valid_n": int(reduced_counts.get(level, 0)),
                "full_training_n": int(full_training_counts.get(level, 0)),
                "reduced_training_n": int(reduced_training_counts.get(level, 0)),
            })
    levels = pd.DataFrame(level_rows)
    save_table(levels, out / "feature_levels.csv")
    support_summary = []
    for col in selected:
        rows = levels.loc[levels.feature == col]
        support_summary.append({
            "feature": col, "type": "numeric" if col in numeric else "categorical",
            "full_valid_observed_n": int(full_valid[col].notna().sum()),
            "reduced_valid_observed_n": int(reduced_valid[col].notna().sum()),
            "full_training_observed_n": int(full_valid.loc[full_train, col].notna().sum()),
            "reduced_training_observed_n": int(reduced_valid.loc[reduced_train, col].notna().sum()),
            "full_valid_levels": int((rows.full_valid_n > 0).sum()),
            "reduced_valid_levels": int((rows.reduced_valid_n > 0).sum()),
            "full_training_levels": int((rows.full_training_n > 0).sum()),
            "reduced_training_levels": int((rows.reduced_training_n > 0).sum()),
            "full_levels_absent_from_reduced_training": int((rows.reduced_training_n == 0).sum()),
            "full_levels_with_under_5_reduced_training_rows": int((rows.reduced_training_n < 5).sum()),
        })
    save_table(pd.DataFrame(support_summary), out / "support_summary.csv")

    median_rows = []
    for cohort, frame, train, test, artifact in (
        ("full", full_valid, full_train, full_test, full_artifact),
        ("reduced", reduced_valid, reduced_train, reduced_test, reduced_artifact),
    ):
        for col in selected:
            if col not in numeric:
                continue
            source_med = float(frame[col].median()) if frame[col].notna().any() else 0.0
            train_frame = frame.loc[train]
            train_med = float(train_frame[col].median()) if train_frame[col].notna().any() else 0.0
            raw_missing = int(frame[col].isna().sum())
            train_missing = int(train_frame[col].isna().sum())
            test_missing = int(frame.loc[test, col].isna().sum())
            public_name = CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.get(col, col)
            indicator = (artifact.get("missing_indicator_map") or {}).get(public_name)
            exported_fills = {
                float(row[public_name])
                for row in artifact["training_feature_rows"]
                if indicator and row[indicator] == "(Missing)"
            }
            if train_missing:
                assert len(exported_fills) == 1
                assert np.isclose(next(iter(exported_fills)), source_med, atol=1e-8)
            changed = not np.isclose(train_med, source_med, atol=1e-10, rtol=0)
            median_rows.append({
                "cohort": cohort, "feature": col, "full_source_median": source_med,
                "training_only_median": train_med, "difference_train_minus_presplit": train_med - source_med,
                "difference_abs": abs(train_med - source_med),
                "exported_training_fill": next(iter(exported_fills)) if exported_fills else None,
                "full_source_missing_n": raw_missing, "training_missing_n": train_missing,
                "test_missing_n": test_missing,
                "rows_with_changed_imputation": raw_missing if changed else 0,
                "training_rows_with_changed_imputation": train_missing if changed else 0,
                "test_rows_with_changed_imputation": test_missing if changed else 0,
                "training_observed_n": int(train_frame[col].notna().sum()),
            })
    medians = pd.DataFrame(median_rows)
    save_table(medians, out / "median_scope.csv")

    # Repeat the original design: draw 1,648 of 32,956 raw rows before dropping
    # rows without a target. No preprocessing or model fitting occurs here.
    rng = np.random.default_rng(20261007)
    sample_size = len(reduced)
    y = full[target].to_numpy(dtype=float)
    missing_matrix = full[selected].isna().to_numpy(dtype=np.int8)
    valid_flag = np.isfinite(y)
    full_missing_rates = full_valid[selected].isna().mean().to_numpy(dtype=float)
    feature_arrays = {col: full[col].to_numpy(dtype=float) for col in numeric}
    full_observed_mean = {col: float(full_valid[col].mean()) for col in numeric}
    full_observed_sd = {col: float(full_valid[col].std(ddof=1)) for col in numeric}
    metrics = ("target_valid_n", "target_mean_difference", "target_smd",
               "target_ks", "max_abs_missingness_pp", "max_abs_numeric_smd")
    simulation_rows = []
    feature_draw_rows = []
    for iteration in range(args.replicates):
        ix = rng.choice(len(full), size=sample_size, replace=False)
        valid_ix = ix[valid_flag[ix]]
        yy = y[valid_ix]
        rates = missing_matrix[valid_ix].mean(axis=0)
        max_num_smd = 0.0
        for col, rate, full_rate in zip(selected, rates, full_missing_rates):
            feature_draw_rows.append({"replicate": iteration + 1, "feature": col,
                                      "metric": "missingness_difference_pp",
                                      "value": float(100 * (rate - full_rate))})
        for col in numeric:
            a = feature_arrays[col][valid_ix]
            observed = a[np.isfinite(a)]
            if len(observed) and full_observed_sd[col]:
                smd = float((observed.mean() - full_observed_mean[col]) / full_observed_sd[col])
                max_num_smd = max(max_num_smd, abs(smd))
                feature_draw_rows.append({"replicate": iteration + 1, "feature": col,
                                          "metric": "observed_mean_smd", "value": smd})
        simulation_rows.append({
            "replicate": iteration + 1,
            "target_valid_n": len(valid_ix),
            "target_mean_difference": float(yy.mean() - full_target.mean()),
            "target_smd": float((yy.mean() - full_target.mean()) / full_target_sd),
            "target_ks": float(ks_2samp(yy, full_target).statistic),
            "max_abs_missingness_pp": float(100 * np.max(np.abs(rates - full_missing_rates))),
            "max_abs_numeric_smd": float(max_num_smd),
        })
    simulation = pd.DataFrame(simulation_rows)
    save_table(simulation, out / "monte_carlo_draws.csv")
    feature_draws = pd.DataFrame(feature_draw_rows)
    save_table(feature_draws, out / "monte_carlo_feature_draws.csv")
    actual_metrics = {
        "target_valid_n": len(reduced_valid),
        "target_mean_difference": target_comparison["mean_difference"],
        "target_smd": target_comparison["standardized_mean_difference"],
        "target_ks": target_comparison["ks_statistic"],
        "max_abs_missingness_pp": float(missing.loc[missing.scope == "target_valid", "difference_pp"].abs().max()),
        "max_abs_numeric_smd": float(features.loc[features.type == "numeric", "standardized_difference"].abs().max()),
    }
    positions = []
    for metric in metrics:
        position = empirical_position(simulation[metric].to_numpy(dtype=float), actual_metrics[metric])
        positions.append({"metric": metric, **position})
    save_table(pd.DataFrame(positions), out / "monte_carlo_positions.csv")
    feature_positions = []
    missing_actual = missing.loc[missing.scope == "target_valid"].set_index("feature")
    numeric_actual = features.loc[features.type == "numeric"].set_index("feature")
    for (feature, metric), rows in feature_draws.groupby(["feature", "metric"]):
        actual = (float(missing_actual.loc[feature, "difference_pp"])
                  if metric == "missingness_difference_pp"
                  else float(numeric_actual.loc[feature, "standardized_difference"]))
        feature_positions.append({"feature": feature, "metric": metric,
                                  **empirical_position(rows.value.to_numpy(dtype=float), actual)})
    save_table(pd.DataFrame(feature_positions), out / "monte_carlo_feature_positions.csv")

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 160})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.histogram_bin_edges(full_target, bins=35)
    ax.hist(full_target, bins=bins, density=True, alpha=.55, label=f"Full (n={len(full_target):,})")
    ax.hist(reduced_target, bins=bins, density=True, alpha=.55, label=f"5% (n={len(reduced_target):,})")
    ax.set(xlabel="German reading competence", ylabel="Density", title="Target distribution, target-valid rows")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "target_distribution.png"); plt.close(fig)

    m = missing.loc[missing.scope == "target_valid"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    yy = np.arange(len(m))
    ax.barh(yy - .18, m.full_missing_pct, height=.35, label="Full")
    ax.barh(yy + .18, m.reduced_missing_pct, height=.35, label="5%")
    ax.set(yticks=yy, yticklabels=m.feature.str.split(" – ").str[-1], xlabel="Missing (%)", title="Predictor missingness, target-valid rows")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "missingness.png"); plt.close(fig)

    f = features.sort_values("standardized_difference")
    labels = [row.feature.split(" – ")[-1] + (f" = {row.level}" if row.level else "") for row in f.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(5, len(f) * .33)))
    ax.barh(np.arange(len(f)), f.standardized_difference)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(yticks=np.arange(len(f)), yticklabels=labels, xlabel="Difference / full cohort SD", title="Predictor standardized differences (observed values)")
    fig.tight_layout(); fig.savefig(out / "standardized_differences.png"); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, metric in zip(axs.flat, metrics):
        values = simulation[metric]
        ax.hist(values, bins=30, color="#7e9bb7")
        ax.axvline(actual_metrics[metric], color="#bf3c36", linewidth=2, label="Actual seed 42")
        ax.set_title(metric.replace("_", " "))
    axs[0, 0].legend(); fig.suptitle("Monte Carlo position of actual 5% sample")
    fig.tight_layout(); fig.savefig(out / "monte_carlo_position.png"); plt.close(fig)

    sparse_names = [
        col for col in selected
        if any(label in col for label in (
            "Wahrgenommener Leistungsdruck", "Computer- und Tabletkompetenz",
            "Unterstützendes Lernklima",
        ))
    ]
    fig, axs = plt.subplots(1, len(sparse_names), figsize=(12, 3.6))
    for ax, col in zip(axs, sparse_names):
        draws = feature_draws.loc[
            (feature_draws.feature == col) & (feature_draws.metric == "observed_mean_smd"), "value"
        ]
        actual = float(numeric_actual.loc[col, "standardized_difference"])
        percentile = float(next(item["percentile_less_or_equal"] for item in feature_positions
                                if item["feature"] == col and item["metric"] == "observed_mean_smd"))
        ax.hist(draws, bins=30, color="#7e9bb7")
        ax.axvline(actual, color="#bf3c36", linewidth=2)
        ax.set(title=f"{col.split(' – ')[-1]}\nactual percentile: {percentile:.1f}",
               xlabel="Observed mean difference / full SD")
    fig.suptitle("Sparse predictor differences across random 5% draws")
    fig.tight_layout(); fig.savefig(out / "monte_carlo_sparse_predictors.png"); plt.close(fig)

    fig, axs = plt.subplots(len(sparse_names), 1, figsize=(10, 8), sharex=True)
    for ax, col in zip(axs, sparse_names):
        rows = levels.loc[levels.feature == col]
        xx = rows.level.astype(float).to_numpy()
        full_share = rows.full_valid_n.to_numpy() / rows.full_valid_n.sum()
        training_share = rows.reduced_training_n.to_numpy() / rows.reduced_training_n.sum()
        ax.plot(xx, full_share, marker="o", label="Full observed share")
        ax.plot(xx, training_share, marker="o", label="Reduced training observed share")
        ax.set(ylabel="Share", title=f"{col.split(' – ')[-1]} ({int(rows.reduced_training_n.sum())} reduced training observed)")
    axs[0].legend(); axs[-1].set_xlabel("Composite item mean")
    fig.suptitle("Observed value support for sparse predictors")
    fig.tight_layout(); fig.savefig(out / "sparse_support.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (cohort, rows) in zip(axs, medians.groupby("cohort", sort=True)):
        rows = rows.iloc[::-1]
        ax.barh(np.arange(len(rows)), rows.difference_train_minus_presplit)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(yticks=np.arange(len(rows)), yticklabels=rows.feature.str.split(" – ").str[-1],
               xlabel="Training-only − pre-split median", title=cohort.capitalize())
    fig.suptitle("Change in numeric fill value when fitted on training rows")
    fig.tight_layout(); fig.savefig(out / "median_scope.png"); plt.close(fig)

    summary = {
        "source_files": {"full": FULL_FILE, "reduced": REDUCED_FILE,
                         "full_export": FULL_ARTIFACT, "reduced_export": REDUCED_ARTIFACT},
        "sampling": {"seed": SEED, "sampled_before_target_filter": True,
                     "raw_population_rows": len(full), "raw_sample_rows": len(reduced),
                     "exact_seed_42_source_row_match": True,
                     "full_target_valid_rows": len(full_valid), "reduced_target_valid_rows": len(reduced_valid),
                     "full_training_rows": len(full_train), "reduced_training_rows": len(reduced_train),
                     "full_test_rows": len(full_test), "reduced_test_rows": len(reduced_test),
                     "full_test_reduced_train_overlap": len(overlap),
                     "corrected_shared_test_rows": len(full_test) - len(overlap)},
        "target_comparison": target_comparison,
        "monte_carlo": {"replicates": args.replicates, "rng_seed": 20261007,
                        "sampling_without_replacement": True,
                        "positions": positions},
        "preprocessing": {"historical_scope": "whole_dataset_before_split",
                          "parameter_comparison": "full source target-valid versus same model's training split",
                          "model_refits_performed": 0},
        "outputs": sorted(
            [x.name for x in out.iterdir()
             if x.is_file() and x.name not in {"REPORT.md", "summary.json"}]
            + ["summary.json"]
        ),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"output_dir": str(out), "sampling": summary["sampling"],
                      "target_comparison": target_comparison,
                      "monte_carlo_positions": positions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
