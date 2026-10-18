"""Find CSV records represented in an exported artifact's held-out test split."""

from io import BytesIO
import math
from typing import Any, Dict

import pandas as pd

from .data_processing import MISSING_CATEGORY_VALUE


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_comparison_csv(
    csv_bytes: bytes,
    column_aliases: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read a comparison CSV using the same header aliases as dataset upload."""
    if not csv_bytes:
        raise ValueError("The CSV file is empty")
    try:
        frame = pd.read_csv(BytesIO(csv_bytes), encoding="utf-8-sig", sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("The CSV must contain at least one data row")
    if column_aliases:
        frame = frame.rename(columns={
            source: destination
            for source, destination in column_aliases.items()
            if source in frame.columns and destination not in frame.columns
        })
    if not frame.columns.is_unique:
        raise ValueError("The CSV has duplicate column names")
    return frame


def compare_csv_with_held_out(
    csv_bytes: bytes,
    artifact: Dict[str, Any],
    *,
    column_aliases: Dict[str, str] | None = None,
    artifact_label: str = "original",
) -> Dict[str, Any]:
    """Match complete model feature rows, including the target when supplied."""
    test_data = artifact.get("held_out_test_data") or {}
    test_rows = test_data.get("feature_rows") or []
    if not test_rows:
        raise ValueError(f"The {artifact_label} artifact has no held-out test rows to check")
    frame = read_comparison_csv(csv_bytes, column_aliases)

    selected = artifact.get("selected_feature_columns") or []
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing model feature columns: {', '.join(missing)}")

    feature_names = artifact.get("feature_names") or selected
    numeric = set(artifact.get("num_features") or [])
    indicators = artifact.get("missing_indicator_map") or {}
    placeholders = artifact.get("numeric_missing_placeholder_values") or {}
    target_name = (artifact.get("metadata") or {}).get("target_column")
    compare_target = bool(target_name and target_name in frame.columns)

    # Pandas turns numeric categorical source columns into strings during training.
    # Match equivalent numeric spellings only when the artifact column is numeric throughout.
    numeric_categories = {
        name for name in feature_names if name not in numeric
        and name not in indicators.values()
        and all(
            value == MISSING_CATEGORY_VALUE or _number(value) is not None
            for value in (row.get(name) for row in test_rows)
        )
    }

    def key(row: Dict[str, Any]) -> tuple[Any, ...]:
        values = []
        for name in feature_names:
            value = row.get(name)
            if name in numeric:
                values.append(_number(value))
            elif name in numeric_categories and value != MISSING_CATEGORY_VALUE:
                values.append(_number(value))
            else:
                values.append(str(value))
        return tuple(values)

    held_out = {}
    sample_ids = test_data.get("sample_ids") or []
    targets = test_data.get("target_values") or []
    for index, row in enumerate(test_rows):
        row_key = key(row)
        if compare_target:
            row_key += (_number(targets[index]),)
        held_out.setdefault(row_key, []).append(index)

    matched_records = []
    matched_test_indices = set()
    for csv_index, csv_row in enumerate(frame.to_dict(orient="records"), start=1):
        model_row = {}
        for name in selected:
            value = csv_row[name]
            missing_value = pd.isna(value)
            if name in numeric:
                number = _number(value)
                if number is None:
                    missing_value = True
                model_row[name] = placeholders.get(name, 0.0) if missing_value else number
                if name in indicators:
                    model_row[indicators[name]] = (
                        MISSING_CATEGORY_VALUE if missing_value else "Observed"
                    )
            else:
                model_row[name] = MISSING_CATEGORY_VALUE if missing_value else str(value)
        row_key = key(model_row)
        if compare_target:
            row_key += (_number(csv_row[target_name]),)
        matched_indices = held_out.get(row_key, [])
        if matched_indices:
            matched_test_indices.update(matched_indices)
            matched_records.append({
                "csv_row": csv_index,
                "held_out_rows": [index + 1 for index in matched_indices],
                "sample_ids": [sample_ids[index] for index in matched_indices] if sample_ids else [],
            })

    return {
        "csv_row_count": len(frame),
        "held_out_row_count": len(test_rows),
        "matched_csv_row_count": len(matched_records),
        "matched_held_out_row_count": len(matched_test_indices),
        "compared_target": compare_target,
        "matched_rows": matched_records,
    }


def compare_csv_after_edited_test_subtraction(
    csv_bytes: bytes,
    original_artifact: Dict[str, Any],
    edited_artifact: Dict[str, Any],
    *,
    column_aliases: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Compare (CSV rows minus edited test rows) against original test rows."""
    original_target = (original_artifact.get("metadata") or {}).get("target_column")
    edited_target = (edited_artifact.get("metadata") or {}).get("target_column")
    if (
        original_target != edited_target
        or set(original_artifact.get("selected_feature_columns") or [])
        != set(edited_artifact.get("selected_feature_columns") or [])
    ):
        raise ValueError("The original and edited artifacts must use the same target and feature columns")

    original = compare_csv_with_held_out(
        csv_bytes, original_artifact, column_aliases=column_aliases,
    )
    edited = compare_csv_with_held_out(
        csv_bytes, edited_artifact, column_aliases=column_aliases,
        artifact_label="edited",
    )
    removed_rows = {match["csv_row"] for match in edited["matched_rows"]}
    matched_rows = [
        match for match in original["matched_rows"]
        if match["csv_row"] not in removed_rows
    ]
    matched_held_out_rows = {
        row_index for match in matched_rows for row_index in match["held_out_rows"]
    }

    return {
        "subtraction_applied": True,
        "csv_row_count": original["csv_row_count"],
        "edited_held_out_row_count": edited["held_out_row_count"],
        "removed_csv_row_count": len(removed_rows),
        "removed_rows": edited["matched_rows"],
        "remaining_csv_row_count": original["csv_row_count"] - len(removed_rows),
        "held_out_row_count": original["held_out_row_count"],
        "original_matches_before_subtraction": original["matched_csv_row_count"],
        "matched_csv_row_count": len(matched_rows),
        "matched_held_out_row_count": len(matched_held_out_rows),
        "compared_target": original["compared_target"],
        "matched_rows": matched_rows,
    }


def compare_csv_to_artifact_test_sets(
    csv_bytes: bytes,
    original_artifact: Dict[str, Any],
    edited_artifact: Dict[str, Any],
    *,
    column_aliases: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Keep the original-only check available when edited test rows are absent."""
    if (edited_artifact.get("held_out_test_data") or {}).get("feature_rows"):
        return compare_csv_after_edited_test_subtraction(
            csv_bytes,
            original_artifact,
            edited_artifact,
            column_aliases=column_aliases,
        )

    direct = compare_csv_with_held_out(
        csv_bytes, original_artifact, column_aliases=column_aliases,
    )
    return {
        **direct,
        "subtraction_applied": False,
        "edited_held_out_row_count": 0,
        "removed_csv_row_count": 0,
        "removed_rows": [],
        "remaining_csv_row_count": direct["csv_row_count"],
        "original_matches_before_subtraction": direct["matched_csv_row_count"],
    }
