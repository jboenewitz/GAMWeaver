import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.training_overlap import _exported_numeric_fills, compare_training_to_original_test


@pytest.fixture
def artifact_pair():
    # The full source split (seed 42, 20% test) contains source rows 8 and 1.
    original = {
        "metadata": {"target_column": "score"},
        "held_out_test_data": {
            "feature_rows": [{"x": 8.0}, {"x": 1.0}],
            "target_values": [108.0, 101.0],
        },
    }
    edited = {
        "metadata": {"target_column": "score"},
        "selected_feature_columns": ["x"],
        "feature_names": ["x"],
        "num_features": ["x"],
        "training_sample_ids": ["0", "1", "2", "3"],
        "training_feature_rows": [
            {"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0},
        ],
        "held_out_test_data": {
            "sample_ids": ["4"],
            "feature_rows": [{"x": 8.0}],
            "target_values": [108.0],
        },
    }
    reduced_csv = b"x,score\n1,101\n2,102\n3,103\n4,104\n8,108\n"
    source_csv = ("x,score\n" + "".join(f"{i},{100 + i}\n" for i in range(10))).encode()
    return original, edited, reduced_csv, source_csv


def test_exact_training_test_overlap_uses_source_rows(artifact_pair):
    original, edited, reduced_csv, source_csv = artifact_pair
    result = compare_training_to_original_test(
        original, edited, reduced_csv, source_csv_bytes=source_csv,
    )

    assert result["status"] == "exact"
    assert result["method"] == "source_rows"
    assert result["training_row_count"] == 4
    assert result["original_test_row_count"] == 2
    assert result["overlap_count"] == 1
    assert result["usable_reduced_csv_rows"] == 5
    shared = result["shared_evaluation"]
    assert shared["original_test_source_ids"] == ["8", "1"]
    assert shared["reduced_training_source_ids"] == ["1", "2", "3", "4"]
    assert shared["kept_original_test_indices"] == [0]
    assert shared["shared_test_row_count"] == 1
    assert set(shared["original_training_source_ids"]).isdisjoint(
        shared["original_test_source_ids"]
    )


def test_target_match_without_source_is_not_claimed_as_exact(artifact_pair):
    original, edited, reduced_csv, _ = artifact_pair
    result = compare_training_to_original_test(original, edited, reduced_csv)

    assert result["status"] == "target_only"
    assert result["overlap_count"] == 1
    assert result["training_row_count"] == 4


def test_shared_namespace_sample_ids_prove_exact_overlap(artifact_pair):
    original, edited, _, _ = artifact_pair
    original["sample_id_namespace"] = "source-v1"
    original["held_out_test_data"]["sample_ids"] = ["source-8", "source-1"]
    edited["sample_id_namespace"] = "source-v1"
    edited["training_sample_ids"] = ["source-1", "source-2", "source-3", "source-4"]

    result = compare_training_to_original_test(original, edited, b"")
    assert result["status"] == "exact"
    assert result["method"] == "shared_sample_ids"
    assert result["overlap_count"] == 1


def test_rejects_reduced_csv_that_does_not_match_training_rows(artifact_pair):
    original, edited, _, source_csv = artifact_pair
    wrong_csv = b"x,score\n9,101\n2,102\n3,103\n4,104\n8,108\n"
    with pytest.raises(ValueError, match="training rows do not match"):
        compare_training_to_original_test(
            original, edited, wrong_csv, source_csv_bytes=source_csv,
        )


def test_rejects_source_csv_that_does_not_match_original_test_split(artifact_pair):
    original, edited, reduced_csv, source_csv = artifact_pair
    wrong_original = copy.deepcopy(original)
    wrong_original["held_out_test_data"]["target_values"] = [108.0, 999.0]
    with pytest.raises(ValueError, match="original artifact test targets do not match"):
        compare_training_to_original_test(
            wrong_original, edited, reduced_csv, source_csv_bytes=source_csv,
        )


def test_source_csv_rejects_two_reduced_artifacts_even_with_shared_namespace(artifact_pair):
    _, edited, reduced_csv, source_csv = artifact_pair
    original = copy.deepcopy(edited)
    original["sample_id_namespace"] = "same-upload"
    edited["sample_id_namespace"] = "same-upload"
    with pytest.raises(ValueError, match="Upload the full-data model export"):
        compare_training_to_original_test(
            original, edited, reduced_csv, source_csv_bytes=source_csv,
        )


def test_rounded_schema_fill_uses_exact_exported_training_value():
    feature = "supportive climate"
    indicator = f"{feature} [missing]"
    artifact = {
        "missing_indicator_map": {feature: indicator},
        "numeric_missing_placeholder_values": {feature: 3.29},
        "training_feature_rows": [{feature: 3.2857142857142856, indicator: "(Missing)"}],
        "held_out_test_data": {
            "feature_rows": [{feature: 3.2857142857142856, indicator: "(Missing)"}],
        },
    }
    assert _exported_numeric_fills(artifact, "Original artifact") == {
        feature: 3.2857142857142856,
    }
    artifact["numeric_missing_placeholder_values"][feature] = 3.28
    with pytest.raises(ValueError, match="fill metadata differs"):
        _exported_numeric_fills(artifact, "Original artifact")
