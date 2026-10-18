"""Verify reduced-model training observations against original-model test rows."""

import hashlib
import math
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from .data_processing import MISSING_CATEGORY_VALUE
from .held_out_overlap import _number, read_comparison_csv


def _row_ids(values: list[Any], valid_index: pd.Index, label: str) -> list[int]:
    result = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} IDs are not CSV row positions") from exc
        if str(index) != str(value) or index not in valid_index:
            raise ValueError(f"{label} IDs do not refer to usable CSV rows")
        result.append(index)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} IDs contain duplicates")
    return result


def _matches_model_row(csv_row: pd.Series, model_row: Dict[str, Any], artifact: Dict[str, Any]) -> bool:
    numeric = set(artifact.get("num_features") or [])
    indicators = artifact.get("missing_indicator_map") or {}
    placeholders = artifact.get("numeric_missing_placeholder_values") or {}
    expected = {}
    for name in artifact.get("selected_feature_columns") or []:
        value = csv_row[name]
        missing = bool(pd.isna(value)) or value == "-"
        if name in numeric:
            number = _number(value)
            missing = missing or number is None
            # Historical exports may omit the exact training median. The
            # explicit missing indicator proves the raw observation was blank.
            expected[name] = (
                model_row.get(name) if missing and name in indicators else
                placeholders.get(name, 0.0) if missing else number
            )
            if name in indicators:
                expected[indicators[name]] = MISSING_CATEGORY_VALUE if missing else "Observed"
        else:
            expected[name] = MISSING_CATEGORY_VALUE if missing else str(value)

    for name in artifact.get("feature_names") or []:
        left, right = expected.get(name), model_row.get(name)
        if name in numeric:
            left_number, right_number = _number(left), _number(right)
            if left_number is None or right_number is None or not math.isclose(
                left_number, right_number, rel_tol=0, abs_tol=1e-7,
            ):
                return False
        elif str(left) != str(right):
            return False
    return True


def _exported_numeric_fills(artifact: Dict[str, Any], label: str) -> Dict[str, float]:
    """Recover the artifact's fitted fills from training rows, without refitting."""
    training_rows = artifact.get("training_feature_rows") or []
    test_rows = (artifact.get("held_out_test_data") or {}).get("feature_rows") or []
    fills = {}
    for feature, indicator in (artifact.get("missing_indicator_map") or {}).items():
        values = {
            float(row[feature]) for row in training_rows
            if row.get(indicator) == MISSING_CATEGORY_VALUE
        }
        if len(values) != 1 or not math.isfinite(next(iter(values))):
            raise ValueError(f"{label} has no unique exported training fill for {feature}")
        fill = values.pop()
        if any(
            not math.isclose(float(row[feature]), fill, rel_tol=0, abs_tol=1e-8)
            for row in test_rows if row.get(indicator) == MISSING_CATEGORY_VALUE
        ):
            raise ValueError(f"{label} test fill differs from its training fill for {feature}")
        fills[feature] = fill
    for feature, declared in (artifact.get("numeric_missing_placeholder_values") or {}).items():
        if feature not in fills:
            raise ValueError(f"{label} fill metadata differs from its training rows")
        declared_value = float(declared)
        exact_value = fills[feature]
        # Older exports use a two-decimal feature-schema default as fallback
        # metadata. The preprocessed training rows retain the fitted median.
        if not (
            math.isclose(declared_value, exact_value, rel_tol=0, abs_tol=1e-8)
            or math.isclose(declared_value, round(exact_value, 2), rel_tol=0, abs_tol=1e-8)
        ):
            raise ValueError(f"{label} fill metadata differs from its training rows")
    return fills


def compare_training_to_original_test(
    original_artifact: Dict[str, Any],
    edited_artifact: Dict[str, Any],
    reduced_csv_bytes: bytes,
    *,
    source_csv_bytes: bytes | None = None,
    column_aliases: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Return exact source-row overlap when IDs or source CSV prove identity.

    With only the reduced CSV, return a uniquely matched target count explicitly
    marked as provisional. Feature values can change between CSV versions, so
    comparing processed feature rows alone cannot prove observation identity.
    """
    original_test = original_artifact.get("held_out_test_data") or {}
    edited_test = edited_artifact.get("held_out_test_data") or {}
    original_targets = original_test.get("target_values") or []
    training_ids = edited_artifact.get("training_sample_ids") or []
    training_rows = edited_artifact.get("training_feature_rows") or []
    original_test_count = len(original_test.get("feature_rows") or [])
    if not original_test_count or len(original_targets) != original_test_count:
        raise ValueError("The original artifact has no complete held-out test set")
    if not training_ids or len(training_ids) != len(training_rows):
        raise ValueError("The edited artifact must include aligned training rows and sample IDs")

    original_ids = original_test.get("sample_ids") or []
    namespace = original_artifact.get("sample_id_namespace")
    if (
        source_csv_bytes is None
        and namespace
        and namespace == edited_artifact.get("sample_id_namespace")
        and len(original_ids) == original_test_count
    ):
        overlap = set(map(str, original_ids)).intersection(map(str, training_ids))
        return {
            "status": "exact",
            "method": "shared_sample_ids",
            "training_row_count": len(training_ids),
            "original_test_row_count": original_test_count,
            "overlap_count": len(overlap),
        }

    frame = read_comparison_csv(reduced_csv_bytes, column_aliases)
    target = (edited_artifact.get("metadata") or {}).get("target_column")
    selected = edited_artifact.get("selected_feature_columns") or []
    missing_columns = [name for name in [target, *selected] if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"The reduced CSV is missing required columns: {', '.join(missing_columns)}")

    target_values = pd.to_numeric(frame[target], errors="coerce")
    valid_index = frame.index[target_values.notna()]
    train_positions = _row_ids(training_ids, valid_index, "Training")
    test_ids = edited_test.get("sample_ids") or []
    test_rows = edited_test.get("feature_rows") or []
    test_targets = edited_test.get("target_values") or []
    if not test_ids or len(test_ids) != len(test_rows) or len(test_ids) != len(test_targets):
        raise ValueError("The edited artifact needs test sample IDs to verify the reduced CSV split")
    test_positions = _row_ids(test_ids, valid_index, "Test")
    if set(train_positions).intersection(test_positions) or (
        set(train_positions).union(test_positions) != set(valid_index)
    ):
        raise ValueError("The edited artifact IDs do not partition the usable reduced CSV rows")
    for position, model_row in zip(train_positions, training_rows):
        if not _matches_model_row(frame.loc[position], model_row, edited_artifact):
            raise ValueError("The edited artifact training rows do not match the reduced CSV")
    for position, model_row, target_value in zip(test_positions, test_rows, test_targets):
        if not _matches_model_row(frame.loc[position], model_row, edited_artifact) or not math.isclose(
            float(target_values.loc[position]), float(target_value), rel_tol=0, abs_tol=1e-8,
        ):
            raise ValueError("The edited artifact test rows do not match the reduced CSV")

    if source_csv_bytes is None:
        training_targets = target_values.loc[train_positions]
        original_test_targets = pd.Series(original_targets, dtype=float)
        if not training_targets.is_unique or not original_test_targets.is_unique:
            raise ValueError("Targets are not unique; upload the original source CSV to verify observation overlap")
        return {
            "status": "target_only",
            "method": "unique_target_values",
            "training_row_count": len(train_positions),
            "original_test_row_count": original_test_count,
            "overlap_count": int(training_targets.isin(original_test_targets).sum()),
            "usable_reduced_csv_rows": len(valid_index),
        }

    source = read_comparison_csv(source_csv_bytes, column_aliases)
    if set(source.columns) != set(frame.columns):
        raise ValueError("The source and reduced CSVs must contain the same columns")
    source = source.loc[:, frame.columns]
    source_targets = pd.to_numeric(source[target], errors="coerce")
    source_valid = source.loc[source_targets.notna()].replace("-", pd.NA)
    reduced_valid = frame.loc[valid_index].replace("-", pd.NA)
    if not source_targets.loc[source_valid.index].is_unique:
        raise ValueError("The source CSV target values must be unique to prove row identity")

    source_hashes = pd.util.hash_pandas_object(source_valid, index=False).to_numpy()
    reduced_hashes = pd.util.hash_pandas_object(reduced_valid, index=False).to_numpy()
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("The source CSV has duplicate complete rows")
    if len(set(reduced_hashes)) != len(reduced_hashes):
        raise ValueError("The reduced CSV has duplicate complete rows, so source identity is ambiguous")
    source_lookup = dict(zip(source_hashes, source_valid.index.to_numpy()))
    if any(value not in source_lookup for value in reduced_hashes):
        raise ValueError("Some usable reduced CSV rows are absent from the source CSV")
    for reduced_position, row_hash in zip(valid_index, reduced_hashes):
        source_position = source_lookup[row_hash]
        if not reduced_valid.loc[reduced_position].equals(source_valid.loc[source_position]):
            raise ValueError("A reduced CSV row does not equal its source row")

    source_train_positions, source_test_positions = train_test_split(
        source_valid.index.to_numpy(), test_size=0.2, random_state=42,
    )
    if len(source_test_positions) != original_test_count:
        raise ValueError(
            f"The left/original artifact has {original_test_count} test rows, but the "
            f"source CSV full-data split has {len(source_test_positions)}. "
            "Upload the full-data model export as the left artifact."
        )
    if any(
        not math.isclose(float(source_targets.loc[position]), float(actual), rel_tol=0, abs_tol=1e-8)
        for position, actual in zip(source_test_positions, original_targets)
    ):
        raise ValueError("The original artifact test targets do not match the source CSV split")
    original_selected = original_artifact.get("selected_feature_columns") or []
    if original_selected:
        for position, model_row in zip(
            source_test_positions, original_test.get("feature_rows") or [],
        ):
            if not _matches_model_row(source.loc[position], model_row, original_artifact):
                raise ValueError("The original artifact test features do not match the source CSV")
    original_training_ids = original_artifact.get("training_sample_ids") or []
    if original_training_ids and all(str(value).isdigit() for value in original_training_ids):
        if original_training_ids != [str(position) for position in source_train_positions]:
            raise ValueError("The original artifact training IDs do not match the source split")
    if original_ids and all(str(value).isdigit() for value in original_ids):
        if original_ids != [str(position) for position in source_test_positions]:
            raise ValueError("The original artifact test IDs do not match the source split")
    reduced_hash_by_index = dict(zip(valid_index, reduced_hashes))
    reduced_train_source_positions = {
        int(source_lookup[reduced_hash_by_index[position]]) for position in train_positions
    }
    overlap_positions = reduced_train_source_positions.intersection(source_test_positions)
    kept_indices = [
        index for index, position in enumerate(source_test_positions)
        if position not in overlap_positions
    ]
    shared_evaluation = None
    if (not original_artifact.get("missing_indicator_map")
            or original_artifact.get("training_feature_rows")):
        shared_evaluation = {
            "source_row_namespace": "iqb-v8-source-row:" + hashlib.sha256(source_csv_bytes).hexdigest()[:16],
            "original_training_source_ids": [str(position) for position in source_train_positions],
            "original_test_source_ids": [str(position) for position in source_test_positions],
            "reduced_training_source_ids": [
                str(source_lookup[reduced_hash_by_index[position]]) for position in train_positions
            ],
            "reduced_test_source_ids": [
                str(source_lookup[reduced_hash_by_index[position]]) for position in test_positions
            ],
            "kept_original_test_indices": kept_indices,
            "shared_test_row_count": len(kept_indices),
            "original_numeric_missing_placeholder_values": _exported_numeric_fills(
                original_artifact, "Original artifact"
            ),
            "reduced_numeric_missing_placeholder_values": _exported_numeric_fills(
                edited_artifact, "Reduced artifact"
            ),
        }
    return {
        "status": "exact",
        "method": "source_rows",
        "training_row_count": len(train_positions),
        "original_test_row_count": original_test_count,
        "overlap_count": len(overlap_positions),
        "usable_reduced_csv_rows": len(valid_index),
        **({"shared_evaluation": shared_evaluation} if shared_evaluation else {}),
    }
