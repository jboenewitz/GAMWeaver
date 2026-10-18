import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.held_out_overlap import (
    compare_csv_after_edited_test_subtraction,
    compare_csv_to_artifact_test_sets,
    compare_csv_with_held_out,
)


@pytest.fixture
def original_artifact():
    return {
        "metadata": {"target_column": "score"},
        "selected_feature_columns": ["age", "group"],
        "feature_names": ["age", "group", "age [missing]"],
        "num_features": ["age"],
        "cat_features": ["group", "age [missing]"],
        "missing_indicator_map": {"age": "age [missing]"},
        "numeric_missing_placeholder_values": {"age": 12.0},
        "held_out_test_data": {
            "feature_rows": [
                {"age": 12.0, "group": "a,b", "age [missing]": "Observed"},
                {"age": 12.0, "group": "x", "age [missing]": "(Missing)"},
            ],
            "target_values": [5.0, 8.0],
            "sample_ids": ["test-1", "test-2"],
        },
    }


def test_overlap_matches_full_rows_and_target_with_quoted_csv(original_artifact):
    csv = b'group,score,age,other\n"a,b",5,12.00,ignored\nx,8,,ignored\n"a,b",6,12,ignored\n'
    result = compare_csv_with_held_out(csv, original_artifact)

    assert result["csv_row_count"] == 3
    assert result["matched_csv_row_count"] == 2
    assert result["matched_held_out_row_count"] == 2
    assert result["compared_target"] is True
    assert result["matched_rows"] == [
        {"csv_row": 1, "held_out_rows": [1], "sample_ids": ["test-1"]},
        {"csv_row": 2, "held_out_rows": [2], "sample_ids": ["test-2"]},
    ]


def test_overlap_can_match_features_without_target_and_counts_duplicate_csv_rows(original_artifact):
    csv = b'age;group\n12;"a,b"\n12;"a,b"\n'
    result = compare_csv_with_held_out(csv, original_artifact)

    assert result["compared_target"] is False
    assert result["matched_csv_row_count"] == 2
    assert result["matched_held_out_row_count"] == 1


def test_overlap_requires_every_model_feature(original_artifact):
    with pytest.raises(ValueError, match="missing model feature columns: group"):
        compare_csv_with_held_out(b"age,score\n12,5\n", original_artifact)


def test_overlap_rejects_artifact_without_test_rows(original_artifact):
    original_artifact["held_out_test_data"] = None
    with pytest.raises(ValueError, match="no held-out test rows"):
        compare_csv_with_held_out(b"age,group\n12,x\n", original_artifact)


def test_overlap_matches_numeric_categorical_spellings(original_artifact):
    original_artifact["held_out_test_data"]["feature_rows"][0]["group"] = "1.0"
    original_artifact["held_out_test_data"]["feature_rows"][1]["group"] = "2.0"
    result = compare_csv_with_held_out(b"age,group,score\n12,1,5\n", original_artifact)
    assert result["matched_csv_row_count"] == 1


def test_overlap_accepts_known_source_column_names(original_artifact):
    result = compare_csv_with_held_out(
        b"Alter,group,score\n12,\"a,b\",5\n",
        original_artifact,
        column_aliases={"Alter": "age"},
    )
    assert result["matched_csv_row_count"] == 1


def test_subtracts_edited_test_matches_before_comparing_to_original(original_artifact):
    edited = copy.deepcopy(original_artifact)
    edited["held_out_test_data"] = {
        "feature_rows": [
            original_artifact["held_out_test_data"]["feature_rows"][0],
            {"age": 22.0, "group": "z", "age [missing]": "Observed"},
        ],
        "target_values": [5.0, 9.0],
        "sample_ids": ["edited-1", "edited-2"],
    }
    csv = (
        b'age,group,score\n12,"a,b",5\n,x,8\n22,z,9\n40,q,1\n12,"a,b",6\n'
    )
    result = compare_csv_after_edited_test_subtraction(
        csv, original_artifact, edited,
    )

    assert result["csv_row_count"] == 5
    assert result["edited_held_out_row_count"] == 2
    assert result["removed_csv_row_count"] == 2
    assert [row["csv_row"] for row in result["removed_rows"]] == [1, 3]
    assert result["remaining_csv_row_count"] == 3
    assert result["original_matches_before_subtraction"] == 2
    assert result["matched_csv_row_count"] == 1
    assert result["matched_held_out_row_count"] == 1
    assert result["matched_rows"] == [
        {"csv_row": 2, "held_out_rows": [2], "sample_ids": ["test-2"]},
    ]


def test_subtraction_removes_all_duplicate_csv_rows_matching_edited_test(original_artifact):
    edited = copy.deepcopy(original_artifact)
    csv = b'age,group\n12,"a,b"\n12,"a,b"\n'
    result = compare_csv_after_edited_test_subtraction(
        csv, original_artifact, edited,
    )

    assert result["compared_target"] is False
    assert result["original_matches_before_subtraction"] == 2
    assert result["removed_csv_row_count"] == 2
    assert result["remaining_csv_row_count"] == 0
    assert result["matched_csv_row_count"] == 0


def test_subtraction_requires_edited_test_rows(original_artifact):
    edited = copy.deepcopy(original_artifact)
    edited["held_out_test_data"] = None
    with pytest.raises(ValueError, match="edited artifact has no held-out test rows"):
        compare_csv_after_edited_test_subtraction(
            b"age,group\n12,x\n", original_artifact, edited,
        )


def test_subtraction_uses_each_artifacts_missing_value_encoding(original_artifact):
    edited = copy.deepcopy(original_artifact)
    edited["numeric_missing_placeholder_values"]["age"] = 99.0
    edited["held_out_test_data"]["feature_rows"][1]["age"] = 99.0
    result = compare_csv_after_edited_test_subtraction(
        b"age,group,score\n,x,8\n", original_artifact, edited,
    )

    assert result["original_matches_before_subtraction"] == 1
    assert result["removed_csv_row_count"] == 1
    assert result["matched_csv_row_count"] == 0


def test_original_only_check_remains_available_without_edited_test_rows(original_artifact):
    edited = copy.deepcopy(original_artifact)
    edited["held_out_test_data"] = None
    result = compare_csv_to_artifact_test_sets(
        b"age,group,score\n,x,8\n", original_artifact, edited,
    )

    assert result["subtraction_applied"] is False
    assert result["removed_csv_row_count"] == 0
    assert result["remaining_csv_row_count"] == 1
    assert result["matched_csv_row_count"] == 1
