"""Build the original-representation IQB study's shared held-out test set.

Trace the reduced training rows to the old full CSV, remove their 219 overlaps
from the full model's 5,249 test rows, and evaluate the three study model
variants on the remaining 5,030 rows. No model or preprocessing is fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.ml_service import (
    CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN,
    MLService,
)
from backend.app.models import ModelComparePreparedArtifact, ModelComparePreviewResponse


CSV_PREFIX = (
    "IQB-BT_2021_student_quest_v1_SUF_Off-site_2606-02a_"
    "german_reading_selected15_v8_de_"
)
FULL_ARTIFACT = "original_model_for_real.json"
REDUCED_REFERENCE = "igann-model-2026-10-07T10-00-24-717427+00-00.json"
REDUCED_ORIGIN = "igann-model-2026-08-13T12-26-00-646301+00-00.json"
GERMAN_EDITED = "final_igann_model_v2.json"
ENGLISH_EDITED = "final_igann_model_v2_english_imputed_no_parentheses_anonymized.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def target_valid_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    return frame.loc[pd.to_numeric(frame[target], errors="coerce").notna()]


def split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(frame.index.to_numpy(), test_size=0.2, random_state=42)


def verify_exported_features(
    source: pd.DataFrame, source_ids: np.ndarray, artifact: dict, label: str,
    exported_rows: list[dict] | None = None,
) -> None:
    """Check old CSV values against exported features without fitting transforms."""
    exported = (exported_rows if exported_rows is not None
                else artifact["held_out_test_data"]["feature_rows"])
    require(len(exported) == len(source_ids), f"{label} exported test size differs")
    numeric = set(artifact["num_features"])
    indicators = artifact.get("missing_indicator_map") or {}
    reverse_names = {
        translated: original
        for original, translated in CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()
    }
    for source_id, model_row in zip(source_ids, exported):
        source_row = source.loc[source_id]
        for feature in artifact["selected_feature_columns"]:
            raw = source_row[reverse_names.get(feature, feature)]
            if feature in numeric:
                number = pd.to_numeric(raw, errors="coerce")
                missing = pd.isna(number)
                if feature in indicators:
                    require(
                        model_row[indicators[feature]] == ("(Missing)" if missing else "Observed"),
                        f"{label} missing indicator disagrees with old CSV at {source_id}: {feature}",
                    )
                if not missing:
                    require(np.isclose(float(model_row[feature]), float(number), atol=1e-8, rtol=0),
                            f"{label} numeric feature disagrees with old CSV at {source_id}: {feature}")
            else:
                expected = "(Missing)" if pd.isna(raw) else str(raw)
                require(model_row[feature] == expected,
                        f"{label} categorical feature disagrees with old CSV at {source_id}: {feature}")


def recover_exported_numeric_placeholders(artifact: dict, label: str) -> dict[str, float]:
    """Read already-fitted imputation constants from the October export rows."""
    training_rows = artifact["training_feature_rows"]
    test_rows = artifact["held_out_test_data"]["feature_rows"]
    recovered = {}
    for feature, indicator in (artifact.get("missing_indicator_map") or {}).items():
        missing_values = {
            float(row[feature]) for row in training_rows if row[indicator] == "(Missing)"
        }
        require(len(missing_values) == 1 and np.isfinite(next(iter(missing_values))),
                f"{label} training rows do not have one exported fill value for {feature}")
        recovered[feature] = missing_values.pop()
        require(all(np.isclose(float(row[feature]), recovered[feature], atol=1e-8, rtol=0)
                    for row in test_rows if row[indicator] == "(Missing)"),
                f"{label} test rows disagree with the exported training fill for {feature}")
    declared = artifact.get("numeric_missing_placeholder_values") or {}
    require(all(feature in recovered and np.isclose(float(value), recovered[feature],
                                                   atol=1e-8, rtol=0)
                for feature, value in declared.items()),
            f"{label} exported fill metadata disagrees with its feature rows")
    return recovered


def verify_model_lineage(
    original_reduced: dict,
    october_reduced: dict,
    german_edited: dict,
    english_edited: dict,
    reduced_csv_name: str,
    reduced_train_targets: pd.Series,
) -> None:
    require(
        original_reduced.get("dataset_name") == reduced_csv_name,
        "The unedited reduced artifact does not name the 5% source CSV",
    )
    require(
        october_reduced.get("dataset_name") == reduced_csv_name
        and october_reduced["selected_feature_columns"]
        == english_edited["selected_feature_columns"]
        and october_reduced["feature_names"] == english_edited["feature_names"]
        and october_reduced["target_column"] == english_edited["target_column"]
        and october_reduced["base_prediction_offset"]
        == original_reduced["base_prediction_offset"],
        "The October reduced export is incompatible with the expert edits",
    )
    require(
        np.isclose(
            float(original_reduced["base_prediction_offset"]),
            float(reduced_train_targets.mean()),
            atol=1e-8,
            rtol=0,
        ),
        "The unedited reduced intercept does not match the 5% training target mean",
    )
    require(
        original_reduced["base_prediction_offset"]
        == german_edited["base_prediction_offset"]
        == english_edited["base_prediction_offset"],
        "The edited artifact does not retain the original reduced intercept",
    )
    original_shapes = {
        item["feature_name"]: (item["x_values"], item["y_values"])
        for item in original_reduced["shape_functions"]
    }
    german_shapes = {
        item["feature_name"]: (item["x_values"], item["y_values"])
        for item in german_edited["shape_functions"]
    }
    require(
        original_shapes == german_shapes,
        "The German edited artifact does not use the same reduced-model base curves",
    )
    require(
        german_edited["exported_at"] == english_edited["exported_at"]
        and german_edited["igann_params"] == english_edited["igann_params"]
        and len(german_edited["shape_functions"])
        == len(english_edited["shape_functions"]),
        "The English edited artifact does not match the German edit export",
    )
    require(
        all(
            german["y_values"] == english["y_values"]
            for german, english in zip(
                german_edited["shape_functions"], english_edited["shape_functions"]
            )
        ),
        "The translated edited artifact changes the base curve contributions",
    )


def build_artifacts(
    source_dir: Path,
    artifact_dir: Path,
) -> tuple[dict, dict, dict]:
    source_full_path = source_dir / (CSV_PREFIX + "random100pct_pv_mean_old.csv")
    reduced_path = source_dir / (CSV_PREFIX + "random5pct_pv_mean.csv")
    source_full = pd.read_csv(source_full_path)
    reduced_csv = pd.read_csv(reduced_path)
    full_artifact = load_json(artifact_dir / FULL_ARTIFACT)
    october_reduced = load_json(artifact_dir / REDUCED_REFERENCE)
    original_reduced = load_json(artifact_dir / REDUCED_ORIGIN)
    german_edited = load_json(artifact_dir / GERMAN_EDITED)
    edited_artifact = load_json(artifact_dir / ENGLISH_EDITED)
    target = full_artifact["target_column"]

    require(source_full.columns.equals(reduced_csv.columns), "The old full and reduced CSV schemas differ")
    source_valid = target_valid_rows(source_full, target)
    reduced_valid = target_valid_rows(reduced_csv, target)
    require(source_valid[target].is_unique, "Source targets are not unique enough to verify row order")
    require(full_artifact.get("dataset_name") == source_full_path.name,
            "The full model export does not name the old 100% CSV")
    full_train_ids, full_test_ids = split_indices(source_valid)
    reduced_train_ids, reduced_test_ids = split_indices(reduced_valid)
    require(len(full_test_ids) == 5249, "Original full-data test size must be 5,249")
    require(len(reduced_train_ids) == 1047, "Reduced-data training size must be 1,047")
    full_targets = source_valid.loc[full_test_ids, target].to_numpy(dtype=float)
    test_data = full_artifact["held_out_test_data"]
    require(
        len(test_data["feature_rows"]) == len(full_test_ids)
        and np.allclose(
            np.asarray(test_data["target_values"], dtype=float),
            full_targets,
            atol=1e-8,
            rtol=0,
        ),
        "The full artifact's test rows do not match the reproduced source split",
    )
    require(test_data.get("sample_ids") == [str(i) for i in full_test_ids],
            "October full export test IDs do not match the original split")
    require(full_artifact.get("training_sample_ids") == [str(i) for i in full_train_ids]
            and len(full_artifact.get("training_feature_rows") or []) == len(full_train_ids),
            "October full export training IDs/rows do not match the original split")
    verify_exported_features(source_valid, full_test_ids, full_artifact, "Full model")
    verify_exported_features(
        source_valid, full_train_ids, full_artifact, "Full training",
        full_artifact["training_feature_rows"],
    )
    full_artifact["numeric_missing_placeholder_values"] = (
        recover_exported_numeric_placeholders(full_artifact, "Full model")
    )
    require(
        np.isclose(
            float(full_artifact["base_prediction_offset"]),
            float(source_valid.loc[full_train_ids, target].mean()),
            atol=1e-8,
            rtol=0,
        ),
        "The full-data intercept does not match the reproduced training split",
    )
    verify_model_lineage(
        original_reduced,
        october_reduced,
        german_edited,
        edited_artifact,
        reduced_path.name,
        reduced_valid.loc[reduced_train_ids, target],
    )

    source_hashes = pd.util.hash_pandas_object(source_valid, index=False).to_numpy()
    reduced_hashes = pd.util.hash_pandas_object(reduced_valid, index=False).to_numpy()
    require(
        len(set(source_hashes)) == len(source_hashes)
        and len(set(reduced_hashes)) == len(reduced_hashes),
        "Source rows must have unique complete-row hashes",
    )
    source_lookup = dict(zip(source_hashes, source_valid.index.to_numpy()))
    require(
        all(value in source_lookup for value in reduced_hashes),
        "The 5% CSV is not an exact subset of the older full CSV",
    )
    reduced_source_ids = dict(
        zip(reduced_valid.index.to_numpy(), (source_lookup[value] for value in reduced_hashes))
    )
    for reduced_index, source_index in reduced_source_ids.items():
        require(
            reduced_valid.loc[reduced_index].equals(source_valid.loc[source_index]),
            "A 5% row hash matches a different source row",
        )
    reduced_train_source_ids = {reduced_source_ids[index] for index in reduced_train_ids}
    overlap = set(full_test_ids).intersection(reduced_train_source_ids)
    require(len(overlap) == 219, "Full-test / reduced-training overlap must be 219")
    shared_positions = [
        position
        for position, source_index in enumerate(full_test_ids)
        if source_index not in reduced_train_source_ids
    ]
    shared_ids = [int(full_test_ids[position]) for position in shared_positions]
    require(len(shared_ids) == 5030, "Shared held-out size must be 5,030")
    require(not set(shared_ids).intersection(full_train_ids), "Shared rows overlap full training")
    require(not set(shared_ids).intersection(reduced_train_source_ids),
            "Shared rows overlap reduced training")
    reverse_names = {
        translated: source
        for source, translated in CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()
    }
    source_features = [
        reverse_names.get(feature, feature)
        for feature in full_artifact["selected_feature_columns"]
    ]
    require(set(source_features).issubset(source_full.columns),
            "The old full CSV lacks selected source features")
    require(october_reduced.get("training_sample_ids") == [str(i) for i in reduced_train_ids],
            "October reduced training IDs do not match the original split")
    reduced_test = october_reduced["held_out_test_data"]
    require(reduced_test.get("sample_ids") == [str(i) for i in reduced_test_ids]
            and np.allclose(reduced_test["target_values"],
                            reduced_valid.loc[reduced_test_ids, target].to_numpy(dtype=float),
                            atol=1e-8, rtol=0),
            "October reduced test IDs/targets do not match the original split")
    verify_exported_features(reduced_valid, reduced_test_ids, october_reduced, "Reduced model")
    require(len(october_reduced.get("training_feature_rows") or []) == 1047,
            "October reduced export lacks its original 1,047 training feature rows")
    verify_exported_features(
        reduced_valid, reduced_train_ids, october_reduced, "Reduced training",
        october_reduced["training_feature_rows"],
    )
    october_reduced["numeric_missing_placeholder_values"] = (
        recover_exported_numeric_placeholders(october_reduced, "Reduced model")
    )
    # Edits are additive point offsets. Keep October base curves and attach only
    # the expert submissions; no model parameters are retrained or refitted.
    october_reduced["shape_function_edits_export"] = edited_artifact["shape_function_edits_export"]
    edited_artifact = october_reduced

    namespace = "iqb-v8-source-row:" + hashlib.sha256(
        source_full_path.read_bytes()
    ).hexdigest()[:16]
    full_artifact["sample_id_namespace"] = namespace
    require(full_artifact.get("preprocessing_scope") == "whole_dataset_before_split"
            and edited_artifact.get("preprocessing_scope") == "whole_dataset_before_split",
            "October exports do not record the historical preprocessing scope")
    # Preserve the October export's already-verified training rows and IDs.
    full_artifact["held_out_test_data"] = {
        "feature_rows": [test_data["feature_rows"][index] for index in shared_positions],
        "target_values": [test_data["target_values"][index] for index in shared_positions],
        "sample_ids": [str(index) for index in shared_ids],
    }
    edited_artifact["sample_id_namespace"] = namespace
    edited_artifact["training_sample_ids"] = [
        str(reduced_source_ids[index]) for index in reduced_train_ids
    ]
    summary = {
        "full_model_export": FULL_ARTIFACT,
        "reduced_model_export": REDUCED_REFERENCE,
        "source_full_csv": source_full_path.name,
        "source_reduced_csv": reduced_path.name,
        "full_source_rows": len(source_valid),
        "reduced_source_rows": len(reduced_valid),
        "full_test_rows": len(full_test_ids),
        "reduced_training_rows": len(reduced_train_ids),
        "excluded_overlap_rows": len(overlap),
        "common_held_out_rows": len(shared_ids),
        "shared_disjoint_from_full_training": True,
        "shared_disjoint_from_reduced_training": True,
        "preprocessing_fitted_on_shared_rows": False,
        "exported_numeric_fill_values": {
            "full": full_artifact["numeric_missing_placeholder_values"],
            "reduced": edited_artifact["numeric_missing_placeholder_values"],
        },
        "ordered_shared_source_ids_sha256": hashlib.sha256(
            json.dumps(shared_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "sample_id_namespace": namespace,
        "source_index_basis": "zero-based row position in the older full CSV",
        "lineage_checks": [
            "October full export training/test IDs, targets, and features match the old 100% split",
            "5% source rows exactly match older full source rows",
            "October reduced export training/test IDs and features match the 5% split",
            "shared source IDs are disjoint from both model training sets",
            "all 5,030 old-representation test rows are retained after overlap removal",
            "expert edit offsets come from the verified German/English export pair",
            "October reduced base curves remain unchanged when expert offsets are attached",
            "October exported training feature rows are reused without fitting preprocessing",
            "model-specific numeric fills come from exported training rows and are checked on exported test rows",
        ],
    }
    return full_artifact, edited_artifact, summary


def evaluate_artifacts(full_artifact: dict, edited_artifact: dict) -> dict:
    """Run the same typed comparison path used by the model-compare endpoint."""
    shared = full_artifact["held_out_test_data"]
    ordered_ids = shared["sample_ids"]
    ordered_targets = shared["target_values"]
    require(len(ordered_ids) == len(set(ordered_ids)) == len(ordered_targets) == 5030,
            "All models require 5,030 unique ordered shared rows and targets")
    require(len(shared["feature_rows"]) == 5030,
            "Shared feature rows and targets differ in length")
    require(full_artifact["sample_id_namespace"] == edited_artifact["sample_id_namespace"],
            "The model source-row namespaces differ")
    require(set(ordered_ids).isdisjoint(full_artifact["training_sample_ids"]),
            "Full-data training rows entered the shared evaluation set")
    require(set(ordered_ids).isdisjoint(edited_artifact["training_sample_ids"]),
            "Reduced-data training rows entered the shared evaluation set")
    # The comparison method receives one ordered feature/target array and uses
    # it for all three predictions. The preprocessing state comes from exports.
    model_row_ids = {model: ordered_ids for model in ("full", "reduced", "expert_adapted")}
    model_targets = {model: ordered_targets for model in model_row_ids}
    require(all(ids == ordered_ids for ids in model_row_ids.values()),
            "The three models would receive different row identities")
    require(all(targets == ordered_targets for targets in model_targets.values()),
            "The three models would receive different target vectors")
    require(full_artifact["preprocessing_scope"] == edited_artifact["preprocessing_scope"]
            == "whole_dataset_before_split", "Historical preprocessing scope changed")
    restore_metadata = MLService._restore_active_dataset_metadata
    auto_load = MLService._auto_load_persisted_dataset
    try:
        MLService._restore_active_dataset_metadata = lambda self: None
        MLService._auto_load_persisted_dataset = lambda self: None
        service = MLService()
    finally:
        MLService._restore_active_dataset_metadata = restore_metadata
        MLService._auto_load_persisted_dataset = auto_load
    full = service.prepare_model_compare_artifact(
        full_artifact, filename="full-data-common-heldout.json"
    )
    edited = service.prepare_model_compare_artifact(
        edited_artifact, filename="edited-reduced-common-heldout.json"
    )
    ModelComparePreparedArtifact(**full)
    ModelComparePreparedArtifact(**edited)
    selected_edits = [
        submission["submission_id"]
        for group in edited["submissions_by_feature"]
        for submission in group["submissions"]
    ]
    preview = service.build_model_compare_preview(
        left_artifact=full,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=selected_edits,
        held_out_test_source="left",
        apply_training_intercept_correction=True,
    )
    ModelComparePreviewResponse(**preview)
    performance = preview["held_out_test_performance"]
    require(
        performance["validation"]["common_holdout_verified"],
        "The generated comparison did not verify both training/test exclusions",
    )
    require(performance["test_size"] == 5030
            and performance["actual_values"] == ordered_targets,
            "The comparison changed the shared 5,030-row target vector")
    require(all(len(performance[key]) == 5030 for key in
                ("baseline_predictions", "reduced_predictions", "edited_predictions")),
            "A model did not predict every ordered shared row")
    require(preview["training_mean_prediction_summary"]["training_row_count"] == 1047
            and np.isclose(
                performance["intercept_correction"]["shift"],
                preview["training_mean_prediction_summary"]["suggested_matching_adjustment"],
                atol=1e-8, rtol=0,
            ), "The fixed intercept correction was not derived from reduced training rows")
    reduced = performance["reduced"]
    edited_metrics = performance["edited"]
    return {
        "test_rows": performance["test_size"],
        "selected_edit_submissions": len(selected_edits),
        "full_data_benchmark": performance["baseline"],
        "original_reduced_data": reduced,
        "edited_reduced_data": edited_metrics,
        "edited_before_intercept_correction": performance[
            "edited_before_intercept_correction"
        ],
        "edited_minus_reduced": {
            key: (
                edited_metrics[key] - reduced[key]
                if edited_metrics[key] is not None and reduced[key] is not None
                else None
            )
            for key in ("mae", "rmse", "r2", "mean_error", "std_dev")
        },
        "relative_mae_change_percent": (
            100 * (edited_metrics["mae"] - reduced["mae"]) / reduced["mae"]
        ),
        "per_observation": {
            key: value
            for key, value in performance["edit_effect"].items()
            if key != "absolute_error_changes"
        },
        "paired_bootstrap": performance["paired_bootstrap"],
        "intercept_correction": performance["intercept_correction"],
        "validation": performance["validation"],
        "ordered_row_identities_identical": True,
        "ordered_shared_source_ids_sha256": hashlib.sha256(
            json.dumps(ordered_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "target_vectors_identical": True,
        "preprocessing_parameters_fitted_on_shared_rows": False,
        "evaluation_set": "5,030 shared held-out observations from the old 100% CSV",
        "caveat": (
            "Historical artifacts fitted missing-value preprocessing before "
            "the train/test split; selected test targets were not used for "
            "model fitting or intercept correction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Generate the validated artifacts without running the comparison metrics",
    )
    args = parser.parse_args()
    full, edited, summary = build_artifacts(args.source_dir, args.artifact_dir)
    performance = None if args.skip_evaluation else evaluate_artifacts(full, edited)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("full-data-common-heldout.json", full),
        ("edited-reduced-common-heldout.json", edited),
        ("split-validation-summary.json", summary),
    ]
    if performance:
        outputs.append(("performance-summary.json", performance))
    for name, payload in outputs:
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if performance:
        print(
            json.dumps(
                {
                    "test_rows": performance["test_rows"],
                    "mae": {
                        "full": performance["full_data_benchmark"]["mae"],
                        "reduced": performance["original_reduced_data"]["mae"],
                        "edited": performance["edited_reduced_data"]["mae"],
                    },
                    "delta_mae_ci_95": performance["paired_bootstrap"]["ci_95"]["mae"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
