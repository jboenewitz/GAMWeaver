"""Complete, read-only dataset diagnostics for the October IQB model lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, ks_2samp, wasserstein_distance
from sklearn.model_selection import train_test_split

from backend.app.ml_service import CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN
from backend.scripts.iqb_sample_audit import FULL_ARTIFACT, FULL_FILE, REDUCED_ARTIFACT, REDUCED_FILE


def save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10g")


def describe(x):
    a = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    q = np.quantile(a, [.01, .05, .25, .5, .75, .95, .99])
    return dict(n=len(a), mean=a.mean(), median=q[3], sd=a.std(ddof=1),
                iqr=q[4]-q[2], minimum=a.min(), maximum=a.max(),
                q01=q[0], q05=q[1], q25=q[2], q50=q[3], q75=q[4],
                q95=q[5], q99=q[6])


def distance(a, b):
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    sd = x.std(ddof=1)
    return dict(smd=(y.mean()-x.mean())/sd if sd else np.nan,
                ks=ks_2samp(x, y).statistic,
                wasserstein=wasserstein_distance(x, y))


def edges_from_full(x):
    # Equal-frequency cuts cannot separate tied discrete values consistently.
    # Deduplicate percentile boundaries and report the effective bin count.
    edges = np.unique(np.quantile(pd.Series(x).dropna(), np.linspace(0, 1, 21)))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def support_counts(values, edges):
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    counts = np.histogram(observed, bins=edges)[0]
    return int((counts == 0).sum()), int(counts[counts > 0].min()) if counts.any() else 0


def pattern_stats(frame, features):
    mask = frame[features].isna()
    patterns = mask.apply(lambda row: "|".join(c for c in features if row[c]), axis=1)
    return mask.sum(axis=1).value_counts().sort_index(), patterns.value_counts()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=1000)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(args.source_dir / FULL_FILE)
    reduced = pd.read_csv(args.source_dir / REDUCED_FILE)
    with (args.artifact_dir / FULL_ARTIFACT).open() as f:
        fa = json.load(f)
    with (args.artifact_dir / REDUCED_ARTIFACT).open() as f:
        ra = json.load(f)
    reverse = {v: k for k, v in CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()}
    selected = [reverse.get(c, c) for c in fa["selected_feature_columns"]]
    numeric = [reverse.get(c, c) for c in fa["num_features"] if reverse.get(c, c) in selected]
    categorical = [c for c in selected if c not in numeric]
    target = fa["target_column"]
    sample_ix = np.sort(np.random.default_rng(42).choice(len(full), size=round(.05*len(full)), replace=False))
    assert full.iloc[sample_ix].reset_index(drop=True).equals(reduced)
    fv = full.loc[full[target].notna()]
    rv = reduced.loc[reduced[target].notna()]
    ft, fte = train_test_split(fv.index.to_numpy(), test_size=.2, random_state=42)
    rt, rte = train_test_split(rv.index.to_numpy(), test_size=.2, random_state=42)
    assert fa["training_sample_ids"] == list(map(str, ft))
    assert ra["training_sample_ids"] == list(map(str, rt))
    assert fa["held_out_test_data"]["sample_ids"] == list(map(str, fte))
    assert ra["held_out_test_data"]["sample_ids"] == list(map(str, rte))
    assert len(set(sample_ix[rt]) & set(fte)) == 219
    cohorts = {"full_valid": fv, "reduced_valid": rv,
               "full_train": fv.loc[ft], "reduced_train": rv.loc[rt]}
    comparisons = [("reduced_valid", "full_valid"),
                   ("reduced_train", "full_valid"),
                   ("reduced_train", "full_train")]

    target_rows, numeric_rows, categorical_rows, tvd_rows = [], [], [], []
    for name, frame in cohorts.items():
        target_rows.append({"cohort": name, **describe(frame[target])})
    for name, ref in comparisons:
        target_rows.append({"cohort": name+"_vs_"+ref, **distance(cohorts[ref][target], cohorts[name][target])})
        for col in numeric:
            a, b = cohorts[ref][col], cohorts[name][col]
            numeric_rows.append({"comparison": name+"_vs_"+ref, "feature": col,
                                 **{"full_"+k: v for k, v in describe(a).items()},
                                 **{"sample_"+k: v for k, v in describe(b).items()},
                                 **distance(a, b),
                                 "full_missing_pct": 100*a.isna().mean(),
                                 "sample_missing_pct": 100*b.isna().mean(),
                                 "missing_difference_pp": 100*(b.isna().mean()-a.isna().mean())})
        for col in categorical:
            a, b = cohorts[ref][col].fillna("(Missing)").astype(str), cohorts[name][col].fillna("(Missing)").astype(str)
            levels = sorted(set(a) | set(b))
            tvd = 0
            for level in levels:
                na, nb = int((a == level).sum()), int((b == level).sum())
                pa, pb = na/len(a), nb/len(b)
                tvd += abs(pb-pa)/2
                categorical_rows.append({"comparison": name+"_vs_"+ref, "feature": col, "level": level,
                                         "full_n": na, "sample_n": nb, "full_pct": 100*pa,
                                         "sample_pct": 100*pb, "difference_pp": 100*(pb-pa),
                                         "abs_difference_pp": 100*abs(pb-pa),
                                         "absent_from_reduced_train": (cohorts["reduced_train"][col].fillna("(Missing)").astype(str) == level).sum() == 0})
            tvd_rows.append({"comparison": name+"_vs_"+ref, "feature": col, "tvd": tvd})
    save(target_rows, out/"target_detailed.csv")
    save(numeric_rows, out/"numeric_detailed.csv")
    save(categorical_rows, out/"categorical_detailed.csv")
    save(tvd_rows, out/"categorical_tvd.csv")

    miss_rows, count_rows, joint_rows = [], [], []
    important = [c for c in numeric if any(s in c for s in ("Leistungsdruck", "Tabletkompetenz", "Lernklima"))]
    for name, frame in cohorts.items():
        for col in selected:
            miss_rows.append({"cohort": name, "feature": col, "missing_n": int(frame[col].isna().sum()),
                              "missing_pct": 100*frame[col].isna().mean()})
        counts, patterns = pattern_stats(frame, selected)
        for n, count in counts.items():
            count_rows.append({"cohort": name, "missing_predictors": n, "n": count, "pct": 100*count/len(frame)})
        for pattern, count in patterns.head(30).items():
            joint_rows.append({"cohort": name, "scope": "all_predictors", "pattern": pattern or "(none)", "n": count, "pct": 100*count/len(frame)})
        _, patterns = pattern_stats(frame, important)
        for pattern, count in patterns.items():
            joint_rows.append({"cohort": name, "scope": "three_item_means", "pattern": pattern or "(none)", "n": count, "pct": 100*count/len(frame)})
    save(miss_rows, out/"missingness_detailed.csv")
    save(count_rows, out/"missing_count_distribution.csv")
    save(joint_rows, out/"joint_missingness.csv")

    leakage = []
    for label, valid, train, test, artifact in (("full", fv, ft, fte, fa), ("reduced", rv, rt, rte, ra)):
        for col in numeric:
            if not valid[col].isna().any():
                continue
            pre, own = valid[col].median(), valid.loc[train, col].median()
            observed = valid[col].dropna()
            stats = describe(observed)
            absdiff = abs(pre-own)
            public = CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.get(col, col)
            indicator = artifact["missing_indicator_map"][public]
            fills = {float(row[public]) for row in artifact["training_feature_rows"] if row[indicator] == "(Missing)"}
            assert len(fills) == 1 and np.isclose(next(iter(fills)), pre)
            leakage.append({"cohort": label, "feature": col, "full_missing_n": int(valid[col].isna().sum()),
                            "full_missing_pct": 100*valid[col].isna().mean(),
                            "train_missing_n": int(valid.loc[train, col].isna().sum()),
                            "train_missing_pct": 100*valid.loc[train, col].isna().mean(),
                            "test_missing_n": int(valid.loc[test, col].isna().sum()),
                            "test_missing_pct": 100*valid.loc[test, col].isna().mean(),
                            "historical_median": pre, "train_only_median": own,
                            "absolute_difference": absdiff,
                            "difference_pct_observed_range": 100*absdiff/(stats["maximum"]-stats["minimum"]),
                            "difference_over_iqr": absdiff/stats["iqr"] if stats["iqr"] else np.nan,
                            "difference_over_sd": absdiff/stats["sd"] if stats["sd"] else np.nan,
                            "affected_test_n": int(valid.loc[test, col].isna().sum()) if absdiff > 1e-10 else 0,
                            "affected_test_pct": 100*valid.loc[test, col].isna().mean() if absdiff > 1e-10 else 0})
    save(leakage, out/"leakage_detailed.csv")

    edge_map = {c: edges_from_full(fv[c]) for c in numeric}
    support = []
    for col in numeric:
        all_stats, train_stats = describe(fv[col]), describe(cohorts["reduced_train"][col])
        edges = edge_map[col]
        empty, minimum = support_counts(cohorts["reduced_train"][col], edges)
        absent = sorted(set(fv[col].dropna().unique())-set(cohorts["reduced_train"][col].dropna().unique()))
        support.append({"feature": col, "full_min": all_stats["minimum"], "full_max": all_stats["maximum"],
                        "reduced_train_min": train_stats["minimum"], "reduced_train_max": train_stats["maximum"],
                        "range_covered_pct": 100*(train_stats["maximum"]-train_stats["minimum"])/(all_stats["maximum"]-all_stats["minimum"]),
                        "full_q01_represented": train_stats["minimum"] <= all_stats["q01"],
                        "full_q99_represented": train_stats["maximum"] >= all_stats["q99"],
                        "effective_quantile_bins": len(edges)-1, "empty_bins": empty,
                        "minimum_count_populated_bin": minimum, "absent_value_count": len(absent),
                        "absent_values": json.dumps([float(v) for v in absent])})
    save(support, out/"support_detailed.csv")
    sparse = []
    for col in categorical:
        counts = cohorts["reduced_train"][col].fillna("(Missing)").astype(str).value_counts()
        for level, n in counts.items():
            sparse.append({"feature": col, "level": level, "train_n": n,
                           "under_5": n < 5, "under_10": n < 10, "under_20": n < 20})
    save(sparse, out/"categorical_train_counts.csv")

    # Conditional on the 1,047 reduced training rows, exchangeability makes
    # overlap with the fixed full test set hypergeometric on target-valid rows.
    hg = hypergeom(len(fv), len(fte), len(rt))
    overlap = {"population": len(fv), "full_test": len(fte), "reduced_train": len(rt),
               "observed": 219, "expected": hg.mean(), "sd": hg.std(),
               "lower_tail_p": hg.cdf(219), "upper_tail_p": hg.sf(218),
               "two_sided_p": min(1, 2*min(hg.cdf(219), hg.sf(218))),
               "central_95_low": int(hg.ppf(.025)), "central_95_high": int(hg.ppf(.975))}

    # Reproduce all three original stages for each alternative sample.
    rng = np.random.default_rng(20261007)
    y = full[target].to_numpy(float)
    valid_flag = np.isfinite(y)
    full_y = fv[target].to_numpy(float)
    full_y_mean, full_y_sd = full_y.mean(), full_y.std(ddof=1)
    numeric_arrays = {c: full[c].to_numpy(float) for c in numeric}
    numeric_reference = {c: (fv[c].mean(), fv[c].std(ddof=1)) for c in numeric}
    missing_matrix = full[selected].isna().to_numpy(np.int8)
    missing_ref = fv[selected].isna().mean().to_numpy(float)
    cat_arrays = {c: full[c].fillna("(Missing)").astype(str).to_numpy() for c in categorical}
    cat_levels = {c: sorted(set(fv[c].fillna("(Missing)").astype(str))) for c in categorical}
    cat_ref = {c: np.array([(fv[c].fillna("(Missing)").astype(str) == v).mean() for v in cat_levels[c]]) for c in categorical}

    def metrics(ix):
        yy = y[ix]
        smds = []
        empty = 0
        for c in numeric:
            v = numeric_arrays[c][ix]
            observed = v[np.isfinite(v)]
            mean, sd = numeric_reference[c]
            smds.append(abs((observed.mean()-mean)/sd) if len(observed) and sd else np.nan)
            empty += support_counts(observed, edge_map[c])[0]
        rates = missing_matrix[ix].mean(axis=0)
        cat_tvd, absent_levels = [], 0
        for c in categorical:
            vals = cat_arrays[c][ix]
            prop = np.array([(vals == v).mean() for v in cat_levels[c]])
            cat_tvd.append(np.abs(prop-cat_ref[c]).sum()/2)
            absent_levels += int((prop == 0).sum())
        return {"training_n": len(ix), "target_mean": yy.mean(), "target_sd": yy.std(ddof=1),
                "target_smd": (yy.mean()-full_y_mean)/full_y_sd,
                "target_ks": ks_2samp(yy, full_y).statistic,
                "max_abs_numeric_smd": np.nanmax(smds), "mean_abs_numeric_smd": np.nanmean(smds),
                "max_abs_missingness_pp": 100*np.abs(rates-missing_ref).max(),
                "mean_abs_missingness_pp": 100*np.abs(rates-missing_ref).mean(),
                "max_categorical_tvd": max(cat_tvd), "missing_full_levels": absent_levels,
                "empty_numeric_bins": empty}

    actual = metrics(sample_ix[rt])
    draws = []
    for j in range(args.replicates):
        ix = np.sort(rng.choice(len(full), size=len(reduced), replace=False))
        ix = ix[valid_flag[ix]]
        train, _ = train_test_split(ix, test_size=.2, random_state=42)
        draws.append({"replicate": j+1, **metrics(train)})
    sim = pd.DataFrame(draws)
    sim.to_csv(out/"monte_carlo_training_draws.csv", index=False, float_format="%.10g")
    positions = []
    for key, value in actual.items():
        values = sim[key].to_numpy(float)
        q025, q05, q95, q975 = np.quantile(values, [.025, .05, .95, .975])
        positions.append({"statistic": key, "actual": value, "mc_median": np.median(values),
                          "mc_q05": q05, "mc_q95": q95, "mc_q025": q025, "mc_q975": q975,
                          "percentile_rank": 100*np.mean(values <= value),
                          "outside_central_95pct": bool(value < q025 or value > q975)})
    save(positions, out/"monte_carlo_training_positions.csv")
    source_rate = len(fv)/len(full)
    sample_rate = len(rv)/len(reduced)
    source_valid_hg = hypergeom(len(full), len(fv), len(reduced))
    summary = {"source_rows": len(full), "source_target_valid": len(fv), "full_train": len(ft),
               "full_test": len(fte), "reduced_source": len(reduced), "reduced_target_valid": len(rv),
               "reduced_train": len(rt), "reduced_test": len(rte), "sampling_seed": 42,
               "split_seed": 42, "monte_carlo_seed": 20261007, "monte_carlo_replicates": args.replicates,
               "source_valid_pct": 100*source_rate, "sample_valid_pct": 100*sample_rate,
               "valid_rate_difference_pp": 100*(sample_rate-source_rate),
               "expected_valid_count": source_valid_hg.mean(), "valid_count_sd": source_valid_hg.std(),
               "valid_count_two_sided_p": min(1, 2*min(source_valid_hg.cdf(len(rv)), source_valid_hg.sf(len(rv)-1))),
               "overlap": overlap}
    (out/"summary_detailed.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
