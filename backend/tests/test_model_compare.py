import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.ml_service import MLService
from backend.app.models import ModelComparePreviewRequest, ModelComparePreviewResponse


class DummyModel:
    def __init__(self) -> None:
        self.boosting_rates = []
        self.linear_model = SimpleNamespace(intercept_=0.0)

    def get_params(self):
        return {}

    def get_shape_functions_as_dict(self):
        return {
            "age": {
                "x": [0.0, 10.0, 20.0, 29.0],
                "y": [0.0, 1.0, 2.0, 3.0],
                "datatype": "numeric",
            },
            "month": {
                "x": ["1", "2", "3"],
                "y": [0.1, 0.2, 0.3],
                "datatype": "categorical",
            },
        }


@pytest.fixture
def ml_service(monkeypatch, tmp_path):
    monkeypatch.setattr(MLService, "_restore_active_dataset_metadata", lambda self: None)
    monkeypatch.setattr(MLService, "_auto_load_persisted_dataset", lambda self: None)

    service = MLService()
    service.active_dataset_file = Path(tmp_path) / "active_dataset.json"
    service.model = DummyModel()
    service.is_trained = True
    service.model_source = "trained"
    service.feature_names = ["age", "month"]
    service.selected_feature_columns = ["age", "month"]
    service.num_features = ["age"]
    service.cat_features = ["month"]
    service.feature_schema = [
        {"name": "age", "feature_type": "numeric", "default_value": 0.0},
        {
            "name": "month",
            "feature_type": "categorical",
            "categorical_options": ["1", "2", "3"],
        },
    ]
    service.feature_schema_map = {
        item["name"]: item for item in service.feature_schema
    }

    dataset = pd.DataFrame(
        {
            "age": list(range(30)),
            "month": [str((idx % 3) + 1) for idx in range(30)],
        }
    )
    service.df = dataset.copy()
    service.X_train = dataset.copy()
    service._build_shape_baseline = lambda: {}
    service._predict_shape_function_sample = (
        lambda sample: float(sample.get("age", sample.get("month", 0.0)))
    )
    return service


def _build_artifact(service, *, target_column="target", selected_feature_columns=None):
    service.target_column = target_column
    artifact = service.export_model_artifact(include_shape_function_edits=False)
    artifact["target_column"] = target_column
    if selected_feature_columns is not None:
        selected_feature_columns = list(selected_feature_columns)
        selected_feature_set = set(selected_feature_columns)
        artifact["selected_feature_columns"] = selected_feature_columns
        artifact["feature_names"] = [
            feature_name
            for feature_name in artifact["feature_names"]
            if feature_name in selected_feature_set
        ]
        artifact["feature_schema"] = [
            item
            for item in artifact["feature_schema"]
            if item.get("name") in selected_feature_set
        ]
        artifact["shape_functions"] = [
            shape_function
            for shape_function in artifact["shape_functions"]
            if shape_function.get("feature_name") in selected_feature_set
        ]
        artifact["cat_features"] = [
            feature_name
            for feature_name in artifact["cat_features"]
            if feature_name in selected_feature_set
        ]
        artifact["num_features"] = [
            feature_name
            for feature_name in artifact["num_features"]
            if feature_name in selected_feature_set
        ]
        if "training_feature_rows" in artifact:
            artifact["training_feature_rows"] = [
                {
                    feature_name: row[feature_name]
                    for feature_name in artifact["feature_names"]
                }
                for row in artifact["training_feature_rows"]
            ]
    artifact.pop("shape_function_edits_export", None)
    return artifact


def _attach_edits_export(artifact, edits_payload):
    artifact["shape_function_edits_export"] = edits_payload
    artifact["artifact_version"] = "1.1"
    return artifact


def _all_submission_ids(prepared_artifact):
    return [
        submission["submission_id"]
        for feature in prepared_artifact["submissions_by_feature"]
        for submission in feature.get("submissions", [])
    ]


def _get_feature_preview(preview, feature_name):
    return next(
        feature
        for feature in preview["feature_previews"]
        if feature["feature_name"] == feature_name
    )


def _set_prepared_shape_values(prepared_artifact, feature_name, *, x_values=None, y_values=None):
    shape_function = next(
        shape_function
        for shape_function in prepared_artifact["shape_functions"]
        if shape_function["feature_name"] == feature_name
    )
    if x_values is not None:
        shape_function["x_values"] = list(x_values)
    if y_values is not None:
        shape_function["y_values"] = list(y_values)


def _uniform_numeric_curve_metrics(reference_x, reference_y, comparison_x, comparison_y):
    grid = np.linspace(
        max(min(reference_x), min(comparison_x)),
        min(max(reference_x), max(comparison_x)),
        100,
    )
    reference = np.interp(grid, reference_x, reference_y)
    comparison = np.interp(grid, comparison_x, comparison_y)
    residuals = reference - comparison
    squared_errors = np.sum(residuals**2)
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r2": float(1 - squared_errors / np.sum((reference - np.mean(reference))**2)),
    }


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_prepare_model_compare_recovers_exact_fill_from_training_rows(
    ml_service, explicit_metadata,
):
    artifact = _build_artifact(ml_service)
    indicator = "age [missing]"
    artifact["feature_names"].append(indicator)
    artifact["cat_features"].append(indicator)
    artifact["missing_indicator_map"] = {"age": indicator}
    artifact["feature_schema"][0]["default_value"] = 3.29
    if explicit_metadata:
        artifact["numeric_missing_placeholder_values"] = {"age": 3.29}
    artifact["shape_functions"].append({
        "feature_name": indicator,
        "feature_type": "categorical",
        "x_values": ["(Missing)", "Observed"],
        "y_values": [1.0, 0.0],
    })
    artifact["training_feature_rows"] = [
        {"age": 3.2857142857142856, "month": "1", indicator: "(Missing)"},
        {"age": 4.0, "month": "2", indicator: "Observed"},
    ]
    artifact.pop("training_sample_ids", None)

    prepared = ml_service.prepare_model_compare_artifact(
        artifact, filename="rounded-fill.json",
    )

    assert prepared["numeric_missing_placeholder_values"]["age"] == (
        3.2857142857142856
    )


def test_prepare_model_compare_accepts_valid_artifacts_without_edits(
    ml_service,
):
    left_artifact = _build_artifact(ml_service, target_column="ridership")
    right_artifact = _build_artifact(ml_service, target_column="ridership")
    left_prepared = ml_service.prepare_model_compare_artifact(
        left_artifact,
        filename="left.json",
    )
    right_prepared = ml_service.prepare_model_compare_artifact(
        right_artifact,
        filename="right.json",
    )
    shared_features = ml_service._validate_model_compare_pair(
        left_prepared,
        right_prepared,
    )
    preview = ml_service.build_model_compare_preview(
        left_artifact=left_prepared,
        right_artifact=right_prepared,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        use_confidence=True,
        feature_names=shared_features,
    )

    assert shared_features == ["age", "month"]
    assert left_prepared["metadata"]["has_edit_export"] is False
    assert right_prepared["metadata"]["has_edit_export"] is False
    for prepared_artifact in (left_prepared, right_prepared):
        for shape_function in prepared_artifact["shape_functions"]:
            source_shape = next(
                source_shape
                for source_shape in left_artifact["shape_functions"]
                if source_shape["feature_name"] == shape_function["feature_name"]
            )
            assert shape_function["y_values"] == pytest.approx(
                source_shape["y_values"]
            )
    assert preview["use_confidence"] is True
    assert len(preview["feature_previews"]) == 2
    assert preview["average_prediction_scores"] == pytest.approx(
        {
            "original": 1.6658653198653195,
            "reduced": 1.6658653198653195,
            "edited": 1.6658653198653195,
            "difference": 0.0,
            "edited_minus_reduced": 0.0,
        }
    )


def test_prepare_model_compare_preserves_non_centered_curve_positions(ml_service):
    artifact = _build_artifact(ml_service, target_column="ridership")
    age_shape = next(
        shape_function
        for shape_function in artifact["shape_functions"]
        if shape_function["feature_name"] == "age"
    )
    age_shape["x_values"] = [1.4, 2.0, 3.0, 4.0]
    age_shape["y_values"] = [5.0, 7.0, 11.0, 17.0]
    age_schema = next(
        feature for feature in artifact["feature_schema"] if feature["name"] == "age"
    )
    age_schema["min_value"] = 1.0
    age_schema["max_value"] = 4.0
    prepared = ml_service.prepare_model_compare_artifact(
        artifact,
        filename="non-centered.json",
    )

    prepared_age = next(
        shape_function
        for shape_function in prepared["shape_functions"]
        if shape_function["feature_name"] == "age"
    )
    assert len(prepared_age["x_values"]) == 30
    assert prepared_age["y_values"][:4] == pytest.approx([5.0] * 4)
    assert prepared_age["y_values"][-1] == pytest.approx(17.0)
    assert sum(prepared_age["y_values"]) != pytest.approx(0.0)

    preview = ml_service.build_model_compare_preview(
        left_artifact=prepared,
        right_artifact=prepared,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        feature_names=["age"],
    )
    age_preview = _get_feature_preview(preview, "age")
    assert age_preview["right_base_y_values"] == pytest.approx(prepared_age["y_values"])
    assert age_preview["right_effective_y_values"] == pytest.approx(
        prepared_age["y_values"]
    )


def test_average_prediction_uses_shared_100_point_numeric_metric_grid(
    ml_service,
):
    """Both prediction means use the per-feature metric grid over the overlap."""
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service, selected_feature_columns=["age"]),
        filename="full-data.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service, selected_feature_columns=["age"]),
        filename="edited.json",
    )
    _set_prepared_shape_values(
        original,
        "age",
        x_values=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        y_values=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    )
    _set_prepared_shape_values(
        edited,
        "age",
        x_values=[10.0, 20.0, 30.0],
        y_values=[0.0, 0.0, 30.0],
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        feature_names=["age"],
    )

    assert preview["average_prediction_scores"] == pytest.approx(
        {
            "original": 20.0,
            "reduced": 250 / 33,
            "edited": 250 / 33,
            "difference": 250 / 33 - 20.0,
            "edited_minus_reduced": 0.0,
        }
    )
    age_contribution = preview["average_prediction_feature_contributions"][
        "uniform"
    ][1]
    assert age_contribution["original"] == pytest.approx(20.0)
    assert age_contribution["edited"] == pytest.approx(250 / 33)


def test_average_prediction_uses_original_artifact_training_rows(ml_service):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    for artifact, age_values in (
        (original, [0.0, 10.0, 20.0, 30.0]),
        (edited, [0.0, 20.0, 40.0, 60.0]),
    ):
        _set_prepared_shape_values(
            artifact,
            "age",
            x_values=[0.0, 10.0, 20.0, 30.0],
            y_values=age_values,
        )
        _set_prepared_shape_values(artifact, "month", y_values=[0.0, 0.0, 0.0])
    ml_service.X_train = pd.DataFrame(
        {"age": [0.0, 10.0], "month": ["1", "1"]}
    )
    ml_service.X_test = pd.DataFrame(
        {"age": [20.0, 30.0], "month": ["2", "2"]}
    )
    original["training_feature_rows"] = [
        {"age": 0.0, "month": "1"},
        {"age": 10.0, "month": "1"},
    ]

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        manual_feature_shifts={"age": 2.0},
        intercept_adjustment=3.0,
    )

    assert preview["main_dataset_average_prediction_scores"] == pytest.approx(
        {
            "original": 5.0,
            "reduced": 10.0,
            "edited": 15.0,
            "difference": 10.0,
            "edited_minus_reduced": 5.0,
        }
    )
    validated = ModelComparePreviewResponse(**preview)
    assert validated.main_dataset_average_prediction_scores.edited == pytest.approx(15.0)
    assert preview["main_dataset_training_row_count"] == 2
    assert preview["main_dataset_average_prediction_shift"] == pytest.approx(2.0)
    without_adjustment = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        manual_feature_shifts={"age": 2.0},
        intercept_adjustment=0.0,
    )
    assert without_adjustment["main_dataset_average_prediction_shift"] == (
        pytest.approx(preview["main_dataset_average_prediction_shift"])
    )
    contributions = preview["average_prediction_feature_contributions"]["main_dataset"]
    assert sum(row["original"] for row in contributions) == pytest.approx(5.0)
    assert sum(row["reduced"] for row in contributions) == pytest.approx(10.0)
    assert sum(row["edited"] for row in contributions) == pytest.approx(15.0)


def test_main_dataset_average_requires_original_artifact_training_rows(ml_service):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    ml_service.is_trained = False
    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
    )
    assert preview["main_dataset_average_prediction_scores"] is not None
    assert preview["main_dataset_training_row_count"] == 30

    ml_service.is_trained = True
    ml_service.X_train = pd.DataFrame({"age": [10.0]})
    original["training_feature_rows"] = None
    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
    )
    assert preview["main_dataset_average_prediction_scores"] is None
    assert preview["main_dataset_average_prediction_shift"] is None


def test_average_prediction_can_use_original_held_out_test_rows(ml_service):
    original_artifact = _build_artifact(ml_service)
    original_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ],
        "target_values": [100.0, 200.0],
    }
    original = ml_service.prepare_model_compare_artifact(
        original_artifact, filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        manual_feature_shifts={"age": 1.0},
        held_out_test_source="right",
    )

    assert preview["test_set_average_prediction_scores"] == pytest.approx(
        {
            "original": 1.65,
            "reduced": 1.65,
            "edited": 2.65,
            "difference": 1.0,
            "edited_minus_reduced": 1.0,
        }
    )
    contributions = preview["average_prediction_feature_contributions"]["test_set"]
    assert sum(row["original"] for row in contributions) == pytest.approx(1.65)
    assert sum(row["reduced"] for row in contributions) == pytest.approx(1.65)
    assert sum(row["edited"] for row in contributions) == pytest.approx(2.65)
    assert preview["held_out_test_performance"] is None


def test_feature_metrics_can_use_original_held_out_test_rows(ml_service):
    original_artifact = _build_artifact(ml_service)
    original_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 0.0, "month": "1"},
            {"age": 0.0, "month": "1"},
            {"age": 0.0, "month": "1"},
            {"age": 15.0, "month": "2"},
        ],
        "target_values": [100.0, 200.0, 300.0, 400.0],
    }
    original = ml_service.prepare_model_compare_artifact(
        original_artifact, filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    _set_prepared_shape_values(edited, "age", y_values=[1.0, 1.0, 2.0, 3.0])
    _set_prepared_shape_values(edited, "month", y_values=[0.4, 0.2, 0.3])
    kwargs = {
        "left_artifact": original,
        "right_artifact": edited,
        "left_selected_submission_ids": [],
        "right_selected_submission_ids": [],
        "manual_feature_shifts": {"age": -1.0, "month": -0.3},
    }
    grid_preview = ml_service.build_model_compare_preview(**kwargs)
    kwargs["feature_metrics_on_original_test_set"] = True
    test_preview = ml_service.build_model_compare_preview(**kwargs)

    age_grid = _get_feature_preview(grid_preview, "age")["comparison_metrics"]
    age_test = _get_feature_preview(test_preview, "age")["comparison_metrics"]
    month_test = _get_feature_preview(test_preview, "month")["comparison_metrics"]
    assert age_grid["baseline_edited_vs_original_base_mae"] == pytest.approx(
        0.1757070707070707
    )
    assert age_test["baseline_edited_vs_original_base_mae"] == pytest.approx(0.75)
    assert age_test["edited_vs_original_base_mae"] == pytest.approx(0.25)
    assert age_test["baseline_edited_vs_original_base_rmse"] == pytest.approx(3**0.5 / 2)
    assert age_test["edited_vs_original_base_rmse"] == pytest.approx(0.5)
    assert age_test["baseline_edited_vs_original_base_r2"] == pytest.approx(-7 / 9)
    assert age_test["edited_vs_original_base_r2"] == pytest.approx(11 / 27)
    assert age_test["edited_vs_original_base_status"] == "decreased"
    assert age_test["edited_vs_original_base_rmse_status"] == "decreased"
    assert age_test["edited_vs_original_base_r2_status"] == "increased"
    assert month_test["baseline_edited_vs_original_base_mae"] == pytest.approx(0.225)
    assert month_test["edited_vs_original_base_mae"] == pytest.approx(0.075)
    assert month_test["baseline_edited_vs_original_base_rmse"] == pytest.approx((0.27 / 4)**0.5)
    assert month_test["edited_vs_original_base_rmse"] == pytest.approx(0.15)
    assert month_test["baseline_edited_vs_original_base_r2"] == pytest.approx(-35.0)
    assert month_test["edited_vs_original_base_r2"] == pytest.approx(-11.0)
    assert test_preview["overall_shape_function_mae"] == grid_preview["overall_shape_function_mae"]
    assert age_test["edited_vs_edited_base_mae"] == age_grid["edited_vs_edited_base_mae"]


def test_numeric_feature_cards_use_100_uniform_curve_points(ml_service):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    x_values = [0.0, 1.0, 9.0, 10.0]
    _set_prepared_shape_values(
        original, "age", x_values=x_values, y_values=[0.0, 0.0, 1.0, 1.0]
    )
    _set_prepared_shape_values(
        edited, "age", x_values=x_values, y_values=[0.0, 0.0, 3.0, 1.0]
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        feature_names=["age"],
        manual_feature_shifts={"age": -0.5},
    )
    metrics = _get_feature_preview(preview, "age")["comparison_metrics"]
    grid = np.linspace(0.0, 10.0, 100)
    reference = np.interp(grid, x_values, [0.0, 0.0, 1.0, 1.0])
    baseline = np.interp(grid, x_values, [0.0, 0.0, 3.0, 1.0])
    edited_values = baseline - 0.5

    def expected_metrics(values):
        residuals = reference - values
        return (
            float(np.mean(np.abs(residuals))),
            float(np.sqrt(np.mean(residuals**2))),
            float(1 - np.sum(residuals**2) / np.sum((reference - np.mean(reference))**2)),
        )

    baseline_mae, baseline_rmse, baseline_r2 = expected_metrics(baseline)
    edited_mae, edited_rmse, edited_r2 = expected_metrics(edited_values)
    assert metrics["baseline_edited_vs_original_base_mae"] == pytest.approx(baseline_mae)
    assert metrics["edited_vs_original_base_mae"] == pytest.approx(edited_mae)
    assert metrics["baseline_edited_vs_original_base_rmse"] == pytest.approx(baseline_rmse)
    assert metrics["edited_vs_original_base_rmse"] == pytest.approx(edited_rmse)
    assert metrics["baseline_edited_vs_original_base_r2"] == pytest.approx(baseline_r2)
    assert metrics["edited_vs_original_base_r2"] == pytest.approx(edited_r2)
    assert preview["overall_shape_function_mae"]["point_count"] == 4
    assert preview["overall_shape_function_mae"]["pooled"]["baseline"] == pytest.approx(0.5)


def test_feature_metrics_test_set_requires_original_feature_rows(ml_service):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )

    with pytest.raises(ValueError, match="Original artifact has no held-out feature rows"):
        ml_service.build_model_compare_preview(
            left_artifact=original,
            right_artifact=edited,
            left_selected_submission_ids=[],
            right_selected_submission_ids=[],
            feature_metrics_on_original_test_set=True,
        )


def test_feature_metrics_on_original_test_rows_include_selected_edits(ml_service):
    original_artifact = _build_artifact(ml_service)
    original_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 0.0, "month": "1"},
            {"age": 15.0, "month": "2"},
        ],
        "target_values": [100.0, 200.0],
    }
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 2.0, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        original_artifact, filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        feature_metrics_on_original_test_set=True,
    )

    metrics = _get_feature_preview(preview, "age")["comparison_metrics"]
    assert metrics["baseline_edited_vs_original_base_mae"] == pytest.approx(0.0)
    assert metrics["edited_vs_original_base_mae"] == pytest.approx(2.0)
    assert metrics["edited_vs_original_base_status"] == "increased"


def test_average_prediction_test_set_option_is_unavailable_without_original_rows(
    ml_service,
):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
    )

    assert preview["test_set_average_prediction_scores"] is None
    assert preview["average_prediction_feature_contributions"]["test_set"] is None


def test_average_prediction_test_set_includes_original_only_internal_features(
    ml_service,
):
    original_artifact = _build_artifact(ml_service)
    original_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ],
        "target_values": [0.0, 0.0],
    }
    original = ml_service.prepare_model_compare_artifact(
        original_artifact, filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    original["feature_names"].append("age_missing")
    original["cat_features"].append("age_missing")
    original["shape_functions"].append(
        {
            "feature_name": "age_missing",
            "feature_type": "categorical",
            "x_values": ["0", "1"],
            "y_values": [0.0, 5.0],
        }
    )
    original["held_out_test_data"]["feature_rows"][0]["age_missing"] = "1"
    original["held_out_test_data"]["feature_rows"][1]["age_missing"] = "0"

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
    )

    scores = preview["test_set_average_prediction_scores"]
    rows = preview["average_prediction_feature_contributions"]["test_set"]
    assert scores == pytest.approx(
        {
            "original": 4.15,
            "reduced": 1.65,
            "edited": 1.65,
            "difference": -2.5,
            "edited_minus_reduced": 0.0,
        }
    )
    assert sum(row["original"] for row in rows) == pytest.approx(scores["original"])
    assert sum(row["edited"] for row in rows) == pytest.approx(scores["edited"])
    assert next(row for row in rows if row["feature_name"] == "age_missing")[
        "is_internal_feature"
    ] is True

    edited["feature_names"].append("age_missing")
    edited["cat_features"].append("age_missing")
    edited["shape_functions"].append(
        {
            "feature_name": "age_missing",
            "feature_type": "categorical",
            "x_values": ["0", "1"],
            "y_values": [0.0, 0.0],
        }
    )
    shifted_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        manual_feature_shifts={"age_missing": 1.0},
    )
    assert shifted_preview["test_set_average_prediction_scores"]["edited"] == (
        pytest.approx(scores["edited"] + 1.0)
    )


def test_average_prediction_ignores_generated_missing_indicator_features(ml_service):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    for artifact, missing_effect in ((original, 10.0), (edited, 30.0)):
        artifact["feature_names"].append("age [missing]")
        artifact["cat_features"].append("age [missing]")
        artifact["missing_indicator_map"] = {"age": "age [missing]"}
        artifact["shape_functions"].append(
            {
                "feature_name": "age [missing]",
                "feature_type": "categorical",
                "x_values": ["(Missing)", "Observed"],
                "y_values": [missing_effect, 0.0],
            }
        )
    original["held_out_test_data"] = {
        "feature_rows": [
            {"age": 10.0, "month": "1", "age [missing]": "(Missing)"},
            {"age": 20.0, "month": "2", "age [missing]": "Observed"},
        ],
        "target_values": [0.0, 0.0],
    }
    ml_service.X_train = pd.DataFrame(
        {"age": [10.0, 20.0], "month": ["1", "2"]}
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        manual_feature_shifts={"age [missing]": 4.0},
    )

    for method, score_key in (
        ("test_set", "test_set_average_prediction_scores"),
        ("main_dataset", "main_dataset_average_prediction_scores"),
    ):
        expected_mean = 1.65 if method == "test_set" else 5 / 3
        assert preview[score_key] == pytest.approx(
            {
                "original": expected_mean,
                "reduced": expected_mean,
                "edited": expected_mean,
                "difference": 0.0,
                "edited_minus_reduced": 0.0,
            }
        )
        rows = preview["average_prediction_feature_contributions"][method]
        assert all(row["feature_name"] != "age [missing]" for row in rows)
        assert sum(row["original"] for row in rows) == pytest.approx(expected_mean)
        assert sum(row["edited"] for row in rows) == pytest.approx(expected_mean)
    assert preview["training_mean_prediction_summary"]["difference_before_correction"] == pytest.approx(0.0)


def test_feature_shift_can_close_the_mean_gap_on_each_average_basis(ml_service):
    original_artifact = _build_artifact(ml_service)
    original_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 20.0, "month": "1"},
            {"age": 25.0, "month": "2"},
        ],
        "target_values": [0.0, 0.0],
    }
    original = ml_service.prepare_model_compare_artifact(
        original_artifact, filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    age_shape = next(
        shape for shape in edited["shape_functions"] if shape["feature_name"] == "age"
    )
    age_shape["y_values"] = [
        value + (2.0 if float(age) >= 15.0 else 0.0)
        for age, value in zip(age_shape["x_values"], age_shape["y_values"])
    ]

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        manual_feature_shifts={"age": 0.5},
    )
    assert preview["average_prediction_scores"]["difference"] != pytest.approx(
        preview["test_set_average_prediction_scores"]["difference"]
    )

    for score_key in (
        "average_prediction_scores",
        "test_set_average_prediction_scores",
    ):
        scores = preview[score_key]
        corrected_shift = 0.5 + scores["original"] - scores["edited"]
        corrected_preview = ml_service.build_model_compare_preview(
            left_artifact=original,
            right_artifact=edited,
            left_selected_submission_ids=[],
            right_selected_submission_ids=[],
            manual_feature_shifts={"age": corrected_shift},
        )
        assert corrected_preview[score_key]["difference"] == pytest.approx(0.0)


def test_prepare_model_compare_returns_submission_tree_for_imported_edits(
    ml_service,
):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice", "profession": "Teacher"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "message": "Lift mid-range ages",
                        "sureness": 8,
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.25, "weight": 0.8},
                            {"x_value": "x:20", "y_value": 0.75, "weight": 0.8},
                        ],
                    }
                ],
            }
        ],
    }
    left_artifact = _attach_edits_export(
        _build_artifact(ml_service, target_column="ridership"),
        edits_payload,
    )
    right_artifact = _build_artifact(ml_service, target_column="ridership")
    left_prepared = ml_service.prepare_model_compare_artifact(
        left_artifact,
        filename="left.json",
    )

    left_age_group = next(
        feature
        for feature in left_prepared["submissions_by_feature"]
        if feature["feature_name"] == "age"
    )
    assert left_prepared["metadata"]["has_edit_export"] is True
    assert left_prepared["metadata"]["edit_submission_count"] == 1
    assert left_age_group["submissions"][0]["submission_id"] == "sub-age"
    assert left_age_group["submissions"][0]["x_summary"] == "10.00, 20.00"


def test_prepare_model_compare_rejects_mismatched_target_column(
    ml_service,
):
    left_artifact = _build_artifact(ml_service, target_column="ridership")
    right_artifact = _build_artifact(ml_service, target_column="demand")
    left_prepared = ml_service.prepare_model_compare_artifact(
        left_artifact,
        filename="left.json",
    )
    right_prepared = ml_service.prepare_model_compare_artifact(
        right_artifact,
        filename="right.json",
    )

    with pytest.raises(ValueError, match="same target_column"):
        ml_service._validate_model_compare_pair(left_prepared, right_prepared)


def test_prepare_model_compare_rejects_mismatched_selected_features(
    ml_service,
):
    left_artifact = _build_artifact(ml_service, target_column="ridership")
    right_artifact = _build_artifact(
        ml_service,
        target_column="ridership",
        selected_feature_columns=["age"],
    )
    left_prepared = ml_service.prepare_model_compare_artifact(
        left_artifact,
        filename="left.json",
    )
    right_prepared = ml_service.prepare_model_compare_artifact(
        right_artifact,
        filename="right.json",
    )

    with pytest.raises(ValueError, match="identical selected_feature_columns"):
        ml_service._validate_model_compare_pair(left_prepared, right_prepared)


def test_preview_weighted_mode_matches_current_weighting_semantics(
    ml_service,
):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}, {"name": "bob"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-a",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 2.0, "weight": 0.2}
                        ],
                    }
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-b",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 0.8}
                        ],
                    }
                ],
            },
        ],
    }
    left_artifact = _attach_edits_export(
        _build_artifact(ml_service, target_column="ridership"),
        edits_payload,
    )
    right_artifact = _build_artifact(ml_service, target_column="ridership")
    prepared = ml_service.prepare_model_compare_artifact(left_artifact, filename="left.json")
    other = ml_service.prepare_model_compare_artifact(right_artifact, filename="right.json")

    preview = ml_service.build_model_compare_preview(
        left_artifact=prepared,
        right_artifact=other,
        left_selected_submission_ids=_all_submission_ids(prepared),
        right_selected_submission_ids=[],
        use_confidence=True,
        feature_names=["age"],
    )

    age_preview = _get_feature_preview(preview, "age")
    assert (
        age_preview["left_effective_y_values"][1]
        - age_preview["left_base_y_values"][1]
    ) == pytest.approx(0.6)
    assert age_preview["right_effective_y_values"][1] == pytest.approx(
        age_preview["right_base_y_values"][1]
    )


def test_preview_can_uniformly_scale_edited_offsets_to_limit_mean_drift(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        use_confidence=True,
        limit_edited_mean_deviation=True,
        feature_names=["age", "month"],
    )

    constraint = preview["mean_deviation_constraint"]
    expected_original_mean = 1.6658653198653195
    expected_scale_factor = 0.1 * expected_original_mean
    assert constraint["applied"] is True
    assert constraint["scale_factor"] == pytest.approx(expected_scale_factor)
    assert constraint["satisfied"] is True
    assert preview["average_prediction_scores"]["original"] == pytest.approx(expected_original_mean)
    assert preview["average_prediction_scores"]["edited"] == pytest.approx(1.1 * expected_original_mean)
    age_preview = _get_feature_preview(preview, "age")
    assert (
        age_preview["right_effective_y_values"][1]
        - age_preview["right_base_y_values"][1]
    ) == pytest.approx(expected_scale_factor)
    assert age_preview["right_individual_edit_traces"][0]["y_values"][1] == pytest.approx(
        age_preview["right_effective_y_values"][1]
    )


def test_preview_mean_limit_uses_only_selected_contribution_features(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        use_confidence=True,
        limit_edited_mean_deviation=True,
        feature_names=["age", "month"],
        mean_deviation_feature_names=["month"],
    )

    constraint = preview["mean_deviation_constraint"]
    age_preview = _get_feature_preview(preview, "age")
    assert constraint["applied"] is False
    assert constraint["scale_factor"] == pytest.approx(1.0)
    assert (
        age_preview["right_effective_y_values"][1]
        - age_preview["right_base_y_values"][1]
    ) == pytest.approx(1.0)


def test_preview_can_limit_only_the_marked_feature_mean_shift(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        use_confidence=True,
        feature_names=["age", "month"],
        mean_deviation_feature_names=["age", "month"],
        mean_deviation_scaled_feature_names=["age"],
    )

    constraint = preview["mean_deviation_constraint"]
    age_preview = _get_feature_preview(preview, "age")
    expected_scale_factor = 0.1 * 1.6658653198653195
    assert constraint["enabled"] is True
    assert constraint["applied"] is True
    assert constraint["feature_scale_factors"] == pytest.approx({"age": expected_scale_factor})
    assert (
        age_preview["right_effective_y_values"][1]
        - age_preview["right_base_y_values"][1]
    ) == pytest.approx(expected_scale_factor)


def test_preview_applies_manual_shift_to_one_edited_feature(ml_service):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="edited.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        feature_names=["age", "month"],
        manual_feature_shifts={"age": 1.0},
    )

    age_preview = _get_feature_preview(preview, "age")
    assert (
        age_preview["right_effective_y_values"][0]
        - age_preview["right_base_y_values"][0]
    ) == pytest.approx(1.0)
    assert preview["average_prediction_scores"]["difference"] == pytest.approx(1.0)


def test_preview_unweighted_mode_ignores_saved_confidence(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}, {"name": "bob"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-a",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 2.0, "weight": 0.2}
                        ],
                    }
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-b",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 0.8}
                        ],
                    }
                ],
            },
        ],
    }
    prepared = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="left.json",
    )
    other = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="right.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=prepared,
        right_artifact=other,
        left_selected_submission_ids=_all_submission_ids(prepared),
        right_selected_submission_ids=[],
        use_confidence=False,
        feature_names=["age"],
    )

    age_preview = _get_feature_preview(preview, "age")
    assert (
        age_preview["left_effective_y_values"][1]
        - age_preview["left_base_y_values"][1]
    ) == pytest.approx(1.5)


def test_preview_deselecting_submission_removes_only_that_contribution(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}, {"name": "bob"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-a",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 2.0, "weight": 0.2}
                        ],
                    }
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-b",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 0.8}
                        ],
                    }
                ],
            },
        ],
    }
    prepared = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="left.json",
    )
    other = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="right.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=prepared,
        right_artifact=other,
        left_selected_submission_ids=["sub-b"],
        right_selected_submission_ids=[],
        use_confidence=True,
        feature_names=["age"],
    )

    age_preview = _get_feature_preview(preview, "age")
    assert (
        age_preview["left_effective_y_values"][1]
        - age_preview["left_base_y_values"][1]
    ) == pytest.approx(0.8)


def test_preview_supports_categorical_feature_submissions(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "month",
                        "feature_type": "categorical",
                        "submission_id": "sub-month",
                        "edited_points": [
                            {"x_value": "2", "y_value": 0.5, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }
    prepared = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="left.json",
    )
    other = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="right.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=prepared,
        right_artifact=other,
        left_selected_submission_ids=["sub-month"],
        right_selected_submission_ids=[],
        use_confidence=True,
        feature_names=["month"],
    )

    month_preview = _get_feature_preview(preview, "month")
    assert (
        month_preview["left_effective_y_values"][1]
        - month_preview["left_base_y_values"][1]
    ) == pytest.approx(0.5)


def test_preview_does_not_mutate_live_runtime_state(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-a",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 0.5}
                        ],
                    }
                ],
            }
        ],
    }
    prepared = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="left.json",
    )
    other = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="right.json",
    )
    before_shape_offsets = dict(ml_service.shape_function_offsets)
    before_selected_features = list(ml_service.selected_feature_columns)
    before_model_source = ml_service.model_source

    preview = ml_service.build_model_compare_preview(
        left_artifact=prepared,
        right_artifact=other,
        left_selected_submission_ids=["sub-a"],
        right_selected_submission_ids=[],
        use_confidence=True,
        feature_names=["age"],
    )

    age_preview = _get_feature_preview(preview, "age")
    assert (
        age_preview["left_effective_y_values"][1]
        - age_preview["left_base_y_values"][1]
    ) == pytest.approx(0.5)
    assert ml_service.shape_function_offsets == before_shape_offsets
    assert ml_service.selected_feature_columns == before_selected_features
    assert ml_service.model_source == before_model_source


def test_preview_returns_curve_metrics_for_edited_artifact(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-right",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 2.0, "weight": 0.5}
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service, target_column="ridership"),
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(
            _build_artifact(ml_service, target_column="ridership"),
            edits_payload,
        ),
        filename="edited.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-right"],
        use_confidence=True,
        feature_names=["age"],
    )

    age_preview = _get_feature_preview(preview, "age")
    assert (
        age_preview["right_effective_y_values"][1]
        - age_preview["right_base_y_values"][1]
    ) == pytest.approx(1.0)
    assert age_preview["comparison_metrics"]["edited_vs_edited_base_mae"] == pytest.approx(1.0)
    assert age_preview["comparison_metrics"]["baseline_edited_vs_edited_base_mae"] == pytest.approx(0.0)
    assert age_preview["comparison_metrics"]["edited_vs_edited_base_status"] == "increased"
    assert age_preview["comparison_metrics"]["edited_vs_original_base_mae"] == pytest.approx(1.0)
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_mae"] == pytest.approx(0.0)
    assert age_preview["comparison_metrics"]["edited_vs_original_base_status"] == "increased"
    assert age_preview["comparison_metrics"]["edited_vs_edited_base_rmse"] == pytest.approx(1.0)
    assert age_preview["comparison_metrics"]["edited_vs_original_base_rmse"] == pytest.approx(1.0)
    assert age_preview["comparison_metrics"]["edited_vs_edited_base_r2"] == pytest.approx(0.2)
    uniform_metrics = _uniform_numeric_curve_metrics(
        age_preview["left_x_values"],
        age_preview["left_base_y_values"],
        age_preview["right_x_values"],
        age_preview["right_effective_y_values"],
    )
    assert age_preview["comparison_metrics"]["edited_vs_original_base_r2"] == pytest.approx(
        uniform_metrics["r2"]
    )
    assert age_preview["comparison_metrics"]["edited_vs_edited_base_r2_status"] == "decreased"
    assert age_preview["comparison_metrics"]["edited_vs_original_base_r2_status"] == "decreased"
    assert preview["average_prediction_scores"] == pytest.approx(
        {
            "original": 1.6658653198653195,
            "reduced": 1.6658653198653195,
            "edited": 2.6658653198653195,
            "difference": 1.0,
            "edited_minus_reduced": 1.0,
        }
    )


def test_preview_mae_status_decreases_when_edits_move_edited_artifact_toward_original(
    ml_service,
):
    original_artifact = _build_artifact(ml_service, target_column="ridership")
    edited_artifact = _build_artifact(ml_service, target_column="ridership")
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-right",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": -1.0, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }

    original = ml_service.prepare_model_compare_artifact(
        original_artifact,
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(edited_artifact, edits_payload),
        filename="edited.json",
    )
    _set_prepared_shape_values(
        original,
        "age",
        y_values=[0.0, 1.0, 2.0, 3.0],
    )
    _set_prepared_shape_values(
        edited,
        "age",
        y_values=[1.0, 2.0, 2.0, 3.0],
    )
    edited_submission = edited["submissions_by_feature"][0]["submissions"][0]
    edited_submission["edited_points"] = [
        {"x_value": "x:0", "y_value": -1.0, "weight": 1.0},
        {"x_value": "x:10", "y_value": -1.0, "weight": 1.0},
        {"x_value": "x:20", "y_value": 0.0, "weight": 1.0},
        {"x_value": "x:29", "y_value": 0.0, "weight": 1.0},
    ]

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-right"],
        use_confidence=True,
        feature_names=["age"],
    )

    age_preview = _get_feature_preview(preview, "age")
    assert age_preview["right_effective_y_values"] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    baseline_uniform_metrics = _uniform_numeric_curve_metrics(
        age_preview["left_x_values"],
        age_preview["left_base_y_values"],
        age_preview["right_x_values"],
        age_preview["right_base_y_values"],
    )
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_mae"] == pytest.approx(
        baseline_uniform_metrics["mae"]
    )
    assert age_preview["comparison_metrics"]["edited_vs_original_base_mae"] == pytest.approx(0.0)
    assert age_preview["comparison_metrics"]["edited_vs_original_base_status"] == "decreased"
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_rmse"] == pytest.approx(
        baseline_uniform_metrics["rmse"]
    )
    assert age_preview["comparison_metrics"]["edited_vs_original_base_rmse"] == pytest.approx(0.0)
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_r2"] == pytest.approx(
        baseline_uniform_metrics["r2"]
    )
    assert age_preview["comparison_metrics"]["edited_vs_original_base_r2"] == pytest.approx(1.0)
    assert age_preview["comparison_metrics"]["edited_vs_original_base_r2_status"] == "increased"


def test_preview_unweighted_mode_changes_edited_curve_metrics(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}, {"name": "bob"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-a",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 2.0, "weight": 0.2}
                        ],
                    }
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-b",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 0.8}
                        ],
                    }
                ],
            },
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )

    weighted_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=_all_submission_ids(edited),
        use_confidence=True,
        feature_names=["age"],
    )
    unweighted_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=_all_submission_ids(edited),
        use_confidence=False,
        feature_names=["age"],
    )

    weighted_age = _get_feature_preview(weighted_preview, "age")
    unweighted_age = _get_feature_preview(unweighted_preview, "age")
    assert weighted_age["right_alternate_confidence_y_values"] == pytest.approx(
        unweighted_age["right_effective_y_values"]
    )
    assert unweighted_age["right_alternate_confidence_y_values"] == pytest.approx(
        weighted_age["right_effective_y_values"]
    )
    assert weighted_age["right_alternate_confidence_y_values"][1] == pytest.approx(2.5)
    assert unweighted_age["right_alternate_confidence_y_values"][1] == pytest.approx(1.6)
    assert ModelComparePreviewResponse.model_validate(weighted_preview).feature_previews[0].right_alternate_confidence_y_values
    assert weighted_age["comparison_metrics"]["edited_vs_edited_base_mae"] == pytest.approx(0.6)
    assert unweighted_age["comparison_metrics"]["edited_vs_edited_base_mae"] == pytest.approx(1.5)
    assert weighted_age["comparison_metrics"]["edited_vs_edited_base_rmse"] == pytest.approx(0.6)
    assert unweighted_age["comparison_metrics"]["edited_vs_edited_base_rmse"] == pytest.approx(1.5)
    assert weighted_age["comparison_metrics"]["edited_vs_edited_base_r2"] == pytest.approx(0.712)
    assert unweighted_age["comparison_metrics"]["edited_vs_edited_base_r2"] == pytest.approx(-0.8)


def test_preview_curve_metrics_change_with_selected_edits(ml_service):
    original_edits_payload = {
        "included": True,
        "users": [{"name": "carol"}],
        "edits": [
            {
                "user_name": "carol",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-original",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 9.0, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}, {"name": "bob"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-a",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 1.0}
                        ],
                    }
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-b",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 3.0, "weight": 1.0}
                        ],
                    }
                ],
            },
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), original_edits_payload),
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )

    no_edits_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        use_confidence=True,
        feature_names=["age"],
    )
    alice_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-a"],
        use_confidence=True,
        feature_names=["age"],
    )
    bob_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-b"],
        use_confidence=True,
        feature_names=["age"],
    )
    original_edit_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=["sub-original"],
        right_selected_submission_ids=[],
        use_confidence=True,
        feature_names=["age"],
    )
    aggregated_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=["sub-original"],
        right_selected_submission_ids=["sub-a", "sub-b"],
        use_confidence=True,
        feature_names=["age"],
    )

    no_edits_metrics = _get_feature_preview(no_edits_preview, "age")["comparison_metrics"]
    alice_metrics = _get_feature_preview(alice_preview, "age")["comparison_metrics"]
    bob_metrics = _get_feature_preview(bob_preview, "age")["comparison_metrics"]
    original_edit_metrics = _get_feature_preview(
        original_edit_preview, "age"
    )["comparison_metrics"]
    aggregated_metrics = _get_feature_preview(aggregated_preview, "age")[
        "comparison_metrics"
    ]

    assert no_edits_metrics["edited_vs_original_base_mae"] == pytest.approx(0.0)
    assert original_edit_metrics["edited_vs_original_base_mae"] == pytest.approx(0.0)
    assert alice_metrics["edited_vs_original_base_mae"] == pytest.approx(1.0)
    assert bob_metrics["edited_vs_original_base_mae"] == pytest.approx(3.0)
    assert aggregated_metrics["edited_vs_original_base_mae"] == pytest.approx(2.0)
    assert alice_metrics["edited_vs_original_base_rmse"] == pytest.approx(1.0)
    assert bob_metrics["edited_vs_original_base_rmse"] == pytest.approx(3.0)
    assert aggregated_metrics["edited_vs_original_base_rmse"] == pytest.approx(2.0)
    left_age = _get_feature_preview(alice_preview, "age")
    reference_grid = np.linspace(
        min(left_age["left_x_values"]), max(left_age["left_x_values"]), 100
    )
    reference_values = np.interp(
        reference_grid, left_age["left_x_values"], left_age["left_base_y_values"]
    )
    reference_variation = np.sum((reference_values - np.mean(reference_values))**2)
    assert alice_metrics["edited_vs_original_base_r2"] == pytest.approx(
        1 - 100 / reference_variation
    )
    assert bob_metrics["edited_vs_original_base_r2"] == pytest.approx(
        1 - 900 / reference_variation
    )
    assert aggregated_metrics["edited_vs_original_base_r2"] == pytest.approx(
        1 - 400 / reference_variation
    )


def test_preview_returns_one_trace_per_selected_edit(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}, {"name": "bob"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-a",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 2.0, "weight": 0.2}
                        ],
                    }
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-b",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 0.8}
                        ],
                    }
                ],
            },
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-a", "sub-b"],
        use_confidence=True,
        feature_names=["age"],
    )

    age_preview = _get_feature_preview(preview, "age")
    assert age_preview["right_effective_y_values"] == pytest.approx(
        [0.6, 1.6, 2.6, 3.6]
    )
    traces_by_id = {
        trace["submission_id"]: trace
        for trace in age_preview["right_individual_edit_traces"]
    }
    assert set(traces_by_id) == {"sub-a", "sub-b"}
    assert traces_by_id["sub-a"]["user_name"] == "alice"
    assert traces_by_id["sub-a"]["sureness"] == 2
    assert traces_by_id["sub-a"]["y_values"] == pytest.approx(
        [0.4, 1.4, 2.4, 3.4]
    )
    assert traces_by_id["sub-b"]["user_name"] == "bob"
    assert traces_by_id["sub-b"]["sureness"] == 8
    assert traces_by_id["sub-b"]["y_values"] == pytest.approx(
        [0.8, 1.8, 2.8, 3.8]
    )

    single_edit_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-a"],
        use_confidence=True,
        feature_names=["age"],
    )
    single_edit_age_preview = _get_feature_preview(single_edit_preview, "age")
    single_edit_trace = single_edit_age_preview["right_individual_edit_traces"][0]

    assert single_edit_trace["original_base_mae"] == pytest.approx(
        single_edit_age_preview["comparison_metrics"]["edited_vs_original_base_mae"]
    )


@pytest.mark.parametrize("use_original_test_rows", [False, True])
def test_single_expert_mae_change_uses_feature_card_sampling(
    ml_service, use_original_test_rows
):
    original_artifact = _build_artifact(ml_service)
    original_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 0.0, "month": "1"},
            {"age": 20.0, "month": "2"},
            {"age": 20.0, "month": "2"},
            {"age": 20.0, "month": "2"},
        ],
        "target_values": [0.0, 0.0, 0.0, 0.0],
    }
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "edited_points": [
                            {"x_value": "x:0", "y_value": 0.0, "weight": 1.0},
                            {"x_value": "x:20", "y_value": -2.0, "weight": 1.0},
                            {"x_value": "x:29", "y_value": 0.0, "weight": 1.0},
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        original_artifact, filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )
    _set_prepared_shape_values(original, "age", y_values=[0.0, 0.0, 1.0, 1.0])
    _set_prepared_shape_values(edited, "age", y_values=[0.0, 0.0, 3.0, 1.0])

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        feature_names=["age"],
        feature_metrics_on_original_test_set=use_original_test_rows,
        manual_feature_shifts={"age": 0.25},
    )
    age_preview = _get_feature_preview(preview, "age")
    trace_mae = age_preview["right_individual_edit_traces"][0]["original_base_mae"]
    card_mae = age_preview["comparison_metrics"]["edited_vs_original_base_mae"]
    assert trace_mae == pytest.approx(card_mae)


def test_preview_uses_expert_prefix_for_anonymized_chart_trace_labels(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "P1"}],
        "edits": [
            {
                "user_name": "P1",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-p1-age",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 2.0, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="edited.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-p1-age"],
        use_confidence=True,
        feature_names=["age"],
    )

    trace = _get_feature_preview(preview, "age")["right_individual_edit_traces"][0]
    assert trace["user_name"] == "E1"


def test_preview_supports_categorical_edited_curve_metrics(ml_service):
    original_artifact = _build_artifact(ml_service, target_column="ridership")
    edited_artifact = _build_artifact(ml_service, target_column="ridership")
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "month",
                        "feature_type": "categorical",
                        "submission_id": "sub-month",
                        "edited_points": [
                            {"x_value": "2", "y_value": -0.2, "weight": 1.0}
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        original_artifact,
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(edited_artifact, edits_payload),
        filename="edited.json",
    )
    _set_prepared_shape_values(
        original,
        "month",
        y_values=[0.1, 0.2, 0.3],
    )
    _set_prepared_shape_values(
        edited,
        "month",
        y_values=[0.1, 0.4, 0.3],
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-month"],
        use_confidence=True,
        feature_names=["month"],
    )

    month_preview = _get_feature_preview(preview, "month")
    assert month_preview["right_effective_y_values"] == pytest.approx(
        [0.1, 0.2, 0.09999999999999998]
    )
    assert month_preview["comparison_metrics"]["baseline_edited_vs_original_base_mae"] == pytest.approx(
        0.06666666666666667
    )
    assert month_preview["comparison_metrics"]["edited_vs_original_base_mae"] == pytest.approx(
        0.06666666666666667
    )
    assert month_preview["comparison_metrics"]["edited_vs_original_base_status"] == "unchanged"
    assert month_preview["comparison_metrics"]["baseline_edited_vs_original_base_rmse"] == pytest.approx(
        0.11547005383792515
    )
    assert month_preview["comparison_metrics"]["edited_vs_original_base_rmse"] == pytest.approx(
        0.11547005383792515
    )
    assert month_preview["comparison_metrics"]["edited_vs_original_base_rmse_status"] == "unchanged"
    assert month_preview["comparison_metrics"]["baseline_edited_vs_original_base_r2"] == pytest.approx(
        -1.0
    )
    assert month_preview["comparison_metrics"]["edited_vs_original_base_r2"] == pytest.approx(
        -1.0
    )
    assert month_preview["comparison_metrics"]["edited_vs_original_base_r2_status"] == "unchanged"


def test_prepare_model_compare_supports_numeric_x_value_mismatches_with_interpolated_mae(
    ml_service,
):
    original_artifact = _build_artifact(ml_service, target_column="ridership")
    edited_artifact = _build_artifact(ml_service, target_column="ridership")
    original = ml_service.prepare_model_compare_artifact(
        original_artifact,
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        edited_artifact,
        filename="edited.json",
    )
    _set_prepared_shape_values(
        original,
        "age",
        y_values=[0.0, 1.0, 2.0, 3.0],
    )
    _set_prepared_shape_values(
        edited,
        "age",
        x_values=[0.0, 20.0, 29.0],
        y_values=[0.0, 2.0, 3.0],
    )
    shared_features = ml_service._validate_model_compare_pair(original, edited)
    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        use_confidence=True,
        feature_names=["age"],
    )
    age_preview = _get_feature_preview(preview, "age")

    assert "age" in shared_features
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_mae"] == pytest.approx(
        0.0
    )
    assert age_preview["comparison_metrics"]["edited_vs_original_base_mae"] == pytest.approx(0.0)
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_rmse"] == pytest.approx(
        0.0
    )
    assert age_preview["comparison_metrics"]["edited_vs_original_base_rmse"] == pytest.approx(0.0)
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_r2"] == pytest.approx(
        1.0
    )
    assert age_preview["comparison_metrics"]["edited_vs_original_base_r2"] == pytest.approx(1.0)


def test_preview_returns_overall_shape_function_mae_for_baseline_and_selected_edits(
    ml_service,
):
    """Overall MAE exposes both point-pooled and equal-feature summaries."""
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "edited_points": [
                            {"x_value": "x:0", "y_value": -1.0, "weight": 1.0},
                            {"x_value": "x:10", "y_value": -1.0, "weight": 1.0},
                            {"x_value": "x:20", "y_value": -1.0, "weight": 1.0},
                            {"x_value": "x:29", "y_value": -1.0, "weight": 1.0},
                        ],
                    },
                    {
                        "feature_name": "month",
                        "feature_type": "categorical",
                        "submission_id": "sub-month",
                        "edited_points": [
                            {"x_value": "1", "y_value": -2.0, "weight": 1.0},
                            {"x_value": "2", "y_value": -2.0, "weight": 1.0},
                            {"x_value": "3", "y_value": -2.0, "weight": 1.0},
                        ],
                    }
                ],
            }
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="full-data.json",
    )
    reduced = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="reduced-data.json",
    )
    _set_prepared_shape_values(original, "age", y_values=[0.0, 0.0, 0.0, 0.0])
    _set_prepared_shape_values(original, "month", y_values=[0.0, 0.0, 0.0])
    _set_prepared_shape_values(reduced, "age", y_values=[1.0, 1.0, 1.0, 1.0])
    _set_prepared_shape_values(reduced, "month", y_values=[3.0, 3.0, 3.0])

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age", "sub-month"],
        feature_names=["age", "month"],
    )

    overall = preview["overall_shape_function_mae"]
    assert overall["feature_count"] == 2
    assert overall["point_count"] == 7
    assert overall["pooled"]["baseline"] == pytest.approx(13 / 7)
    assert overall["pooled"]["edited"] == pytest.approx(3 / 7)
    assert overall["pooled"]["status"] == "decreased"
    assert overall["mean_per_feature"]["baseline"] == pytest.approx(2.0)
    assert overall["mean_per_feature"]["edited"] == pytest.approx(0.5)
    assert overall["mean_per_feature"]["status"] == "decreased"


def test_preview_reports_each_experts_complete_model_mae_with_confidence_setting(
    ml_service,
):
    """Each expert MAE includes that expert's edits across every feature."""
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}, {"name": "bob"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "alice-age",
                        "edited_points": [
                            {"x_value": f"x:{value}", "y_value": 2.0, "weight": 0.2}
                            for value in (0, 10, 20, 29)
                        ],
                    },
                    {
                        "feature_name": "month",
                        "feature_type": "categorical",
                        "submission_id": "alice-month",
                        "edited_points": [
                            {"x_value": str(value), "y_value": 4.0, "weight": 0.5}
                            for value in (1, 2, 3)
                        ],
                    },
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "bob-age",
                        "edited_points": [
                            {"x_value": f"x:{value}", "y_value": 1.0, "weight": 0.8}
                            for value in (0, 10, 20, 29)
                        ],
                    }
                ],
            },
        ],
    }
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="full-data.json",
    )
    reduced = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="reduced-data.json",
    )
    for artifact in (original, reduced):
        _set_prepared_shape_values(artifact, "age", y_values=[0.0] * 4)
        _set_prepared_shape_values(artifact, "month", y_values=[0.0] * 3)

    weighted_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["alice-age", "alice-month", "bob-age"],
        use_confidence=True,
        feature_names=["age", "month"],
    )
    unweighted_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["alice-age", "alice-month", "bob-age"],
        use_confidence=False,
        feature_names=["age", "month"],
    )
    filtered_overall_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["alice-age", "alice-month", "bob-age"],
        use_confidence=False,
        feature_names=["age", "month"],
        overall_mae_feature_names=["age"],
    )
    no_selected_experts_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        use_confidence=False,
        feature_names=["age", "month"],
    )
    single_feature_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        use_confidence=False,
        feature_names=["age"],
    )

    static_changes = weighted_preview["all_feature_mae_change"]
    assert {
        item["feature_name"]: item["mae"] - item["baseline_mae"]
        for item in static_changes["without_confidence"]
    } == pytest.approx({"age": 1.5, "month": 4.0})
    assert {
        item["feature_name"]: item["mae"] - item["baseline_mae"]
        for item in static_changes["with_confidence"]
    } == pytest.approx({"age": 0.6, "month": 2.0})
    assert static_changes == unweighted_preview["all_feature_mae_change"]
    assert static_changes == filtered_overall_preview["all_feature_mae_change"]
    assert static_changes == no_selected_experts_preview["all_feature_mae_change"]
    assert static_changes == single_feature_preview["all_feature_mae_change"]
    serialized_preview = ModelComparePreviewResponse(**weighted_preview).model_dump()
    assert serialized_preview["all_feature_mae_change"] == static_changes

    weighted_age_traces = {
        trace["user_name"]: trace
        for trace in _get_feature_preview(weighted_preview, "age")["right_individual_edit_traces"]
    }
    unweighted_age_traces = {
        trace["user_name"]: trace
        for trace in _get_feature_preview(unweighted_preview, "age")["right_individual_edit_traces"]
    }
    for name, weighted_mae, unweighted_mae in (
        ("alice", 0.4, 2.0),
        ("bob", 0.8, 1.0),
    ):
        assert weighted_age_traces[name]["original_base_mae_with_confidence"] == pytest.approx(weighted_mae)
        assert weighted_age_traces[name]["original_base_mae_without_confidence"] == pytest.approx(unweighted_mae)
        assert unweighted_age_traces[name]["original_base_mae_with_confidence"] == pytest.approx(weighted_mae)
        assert unweighted_age_traces[name]["original_base_mae_without_confidence"] == pytest.approx(unweighted_mae)
    assert ModelComparePreviewResponse(**weighted_preview).model_dump()["feature_previews"][0][
        "right_individual_edit_traces"
    ][0]["original_base_mae_with_confidence"] is not None

    weighted_mae = {
        metric["user_name"]: metric["mae"]
        for metric in weighted_preview["overall_shape_function_mae"][
            "expert_complete_model_mae"
        ]
    }
    unweighted_mae = {
        metric["user_name"]: metric["mae"]
        for metric in unweighted_preview["overall_shape_function_mae"][
            "expert_complete_model_mae"
        ]
    }

    assert weighted_mae == pytest.approx({"alice": 7.6 / 7, "bob": 3.2 / 7})
    assert unweighted_mae == pytest.approx({"alice": 20 / 7, "bob": 4 / 7})
    assert {
        metric["user_name"]: metric["baseline_mae"]
        for metric in weighted_preview["overall_shape_function_mae"][
            "expert_complete_model_mae"
        ]
    } == pytest.approx({"alice": 0.0, "bob": 0.0})
    assert {
        metric["user_name"]: metric["status"]
        for metric in weighted_preview["overall_shape_function_mae"][
            "expert_complete_model_mae"
        ]
    } == {"alice": "increased", "bob": "increased"}
    assert filtered_overall_preview["overall_shape_function_mae"]["feature_count"] == 1
    assert {
        metric["user_name"]: metric["mae"]
        for metric in filtered_overall_preview["overall_shape_function_mae"][
            "expert_complete_model_mae"
        ]
    } == pytest.approx(unweighted_mae)


def test_preview_limits_overall_shape_function_mae_to_selected_features(ml_service):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="full-data.json",
    )
    reduced = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="reduced-data.json",
    )
    _set_prepared_shape_values(original, "age", y_values=[0.0, 0.0, 0.0, 0.0])
    _set_prepared_shape_values(original, "month", y_values=[0.0, 0.0, 0.0])
    _set_prepared_shape_values(reduced, "age", y_values=[1.0, 1.0, 1.0, 1.0])
    _set_prepared_shape_values(reduced, "month", y_values=[3.0, 3.0, 3.0])

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        feature_names=["age", "month"],
        overall_mae_feature_names=["month"],
    )

    overall = preview["overall_shape_function_mae"]
    assert overall["feature_count"] == 1
    assert overall["point_count"] == 3
    assert overall["pooled"]["baseline"] == pytest.approx(3.0)
    assert overall["pooled"]["edited"] == pytest.approx(3.0)


def test_preview_compares_both_artifacts_on_the_reduced_artifacts_held_out_test_set(
    ml_service,
):
    """The selected test rows are shared by the original and edited models."""
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 1.0},
                            {"x_value": "x:20", "y_value": 1.0, "weight": 1.0},
                        ],
                    }
                ],
            }
        ],
    }
    full_artifact = _build_artifact(ml_service)
    reduced_artifact = _attach_edits_export(
        _build_artifact(ml_service),
        edits_payload,
    )
    reduced_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ],
        "target_values": [1.1, 2.2],
    }
    full = ml_service.prepare_model_compare_artifact(
        full_artifact,
        filename="full-data.json",
    )
    reduced = ml_service.prepare_model_compare_artifact(
        reduced_artifact,
        filename="reduced-data.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=full,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        feature_names=["age", "month"],
    )

    held_out = preview["held_out_test_performance"]
    assert held_out["test_size"] == 2
    assert held_out["baseline"]["mae"] == pytest.approx(0.0)
    assert held_out["baseline"]["rmse"] == pytest.approx(0.0)
    assert held_out["baseline"]["std_dev"] == pytest.approx(0.55)
    assert held_out["baseline"]["r2"] == pytest.approx(1.0)
    assert held_out["baseline"]["mean_error"] == pytest.approx(0.0)
    assert held_out["edited"]["mae"] == pytest.approx(1.0)
    assert held_out["edited"]["rmse"] == pytest.approx(1.0)
    assert held_out["edited"]["std_dev"] == pytest.approx(0.55)
    assert held_out["edited"]["r2"] == pytest.approx(1 - 2 / 0.605)
    assert held_out["edited"]["mean_error"] == pytest.approx(1.0)
    assert held_out["actual_values"] == pytest.approx([1.1, 2.2])
    assert held_out["baseline_predictions"] == pytest.approx([1.1, 2.2])
    assert held_out["reduced_predictions"] == pytest.approx([1.1, 2.2])
    assert held_out["reduced"]["mae"] == pytest.approx(0.0)
    assert held_out["reduced"]["std_dev"] == pytest.approx(0.55)
    assert held_out["edited_predictions"] == pytest.approx([2.1, 3.2])
    assert held_out["edit_effect"]["worsened_percent"] == pytest.approx(100.0)
    assert held_out["edit_effect"]["median_absolute_error_change"] == pytest.approx(1.0)
    assert held_out["paired_bootstrap"]["ci_95"]["mae"] == pytest.approx([1.0, 1.0])


def test_training_only_correction_cancels_edit_shift_without_using_test_targets(
    ml_service,
):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [{"user_name": "alice", "shape_functions": [{
            "feature_name": "age", "feature_type": "numeric",
            "submission_id": "sub-age",
            "edited_points": [
                {"x_value": "x:10", "y_value": 1.0, "weight": 1.0},
                {"x_value": "x:20", "y_value": 1.0, "weight": 1.0},
            ],
        }]}],
    }
    full_artifact = _build_artifact(ml_service)
    full_artifact["held_out_test_data"] = {
        "feature_rows": [{"age": 10.0, "month": "1"}, {"age": 20.0, "month": "2"}],
        "target_values": [100.0, 200.0],
    }
    reduced_artifact = _attach_edits_export(_build_artifact(ml_service), edits_payload)
    full = ml_service.prepare_model_compare_artifact(full_artifact, filename="full.json")
    reduced = ml_service.prepare_model_compare_artifact(reduced_artifact, filename="reduced.json")
    options = dict(
        left_artifact=full, right_artifact=reduced,
        left_selected_submission_ids=[], right_selected_submission_ids=["sub-age"],
        held_out_test_source="left", apply_training_intercept_correction=True,
    )
    first = ml_service.build_model_compare_preview(**options)["held_out_test_performance"]
    full["held_out_test_data"]["target_values"] = [-100.0, -200.0]
    second = ml_service.build_model_compare_preview(**options)["held_out_test_performance"]
    assert first["intercept_correction"] == {
        "enabled": True, "shift": pytest.approx(-1.0),
        "calculated_on": "reduced training rows",
    }
    assert first["edited_predictions"] == pytest.approx(first["reduced_predictions"])
    assert first["edited_before_intercept_predictions"] == pytest.approx(
        [value + 1.0 for value in first["reduced_predictions"]]
    )
    assert first["edited_before_intercept_correction"]["mae"] != first["edited"]["mae"]
    assert second["intercept_correction"] == first["intercept_correction"]
    assert second["edited_predictions"] == pytest.approx(first["edited_predictions"])


def test_shared_test_rows_use_each_models_numeric_missing_placeholder(ml_service):
    full = {
        "base_prediction_offset": 0.0,
        "feature_names": ["age", "age [missing]"],
        "cat_features": ["age [missing]"],
        "missing_indicator_map": {"age": "age [missing]"},
        "numeric_missing_placeholder_values": {"age": 11.0},
        "shape_functions": [
            {"feature_name": "age", "feature_type": "numeric", "x_values": [0, 100], "y_values": [0, 100]},
            {"feature_name": "age [missing]", "feature_type": "categorical", "x_values": ["Observed", "(Missing)"], "y_values": [0, 0]},
        ],
        "held_out_test_data": {
            "feature_rows": [{"age": 10.0, "age [missing]": "(Missing)"}],
            "target_values": [20.0],
        },
    }
    reduced = copy.deepcopy(full)
    reduced["numeric_missing_placeholder_values"] = {"age": 20.0}
    reduced["training_feature_rows"] = [{"age": 20.0, "age [missing]": "(Missing)"}]
    result = ml_service._build_held_out_test_performance(
        original_artifact=full, edited_artifact=reduced,
        test_data_artifact=full, test_data_source="left",
        edited_offsets={}, edited_feature_shifts={},
    )
    assert result["baseline_predictions"] == [10.0]
    assert result["reduced_predictions"] == [20.0]
    assert result["validation"]["reduced_training_feature_matches"] == 1


def test_held_out_evaluation_requires_disjoint_source_ids_when_provided(ml_service):
    artifact = {
        "base_prediction_offset": 0.0,
        "feature_names": ["age"],
        "shape_functions": [{"feature_name": "age", "feature_type": "numeric", "x_values": [0, 20], "y_values": [0, 20]}],
        "sample_id_namespace": "shared-source",
        "training_sample_ids": ["train-1"],
        "held_out_test_data": {
            "feature_rows": [{"age": 10.0}],
            "target_values": [10.0],
            "sample_ids": ["test-1"],
        },
    }
    reduced = copy.deepcopy(artifact)
    reduced["training_sample_ids"] = ["test-1"]
    options = dict(
        original_artifact=artifact, edited_artifact=reduced,
        test_data_artifact=artifact, test_data_source="left",
        edited_offsets={}, edited_feature_shifts={},
    )
    with pytest.raises(ValueError, match="Reduced-data training data contains 1 selected test sample IDs"):
        ml_service._build_held_out_test_performance(**options)
    reduced["training_sample_ids"] = ["train-2"]
    result = ml_service._build_held_out_test_performance(**options)
    assert result["validation"]["common_holdout_verified"] is True


def test_intercept_adjustment_changes_only_edited_prediction_base(ml_service):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [{
            "user_name": "alice",
            "shape_functions": [{
                "feature_name": "age",
                "feature_type": "numeric",
                "submission_id": "sub-age",
                "edited_points": [
                    {"x_value": "x:10", "y_value": 1.0, "weight": 1.0},
                    {"x_value": "x:20", "y_value": 1.0, "weight": 1.0},
                ],
            }],
        }],
    }
    held_out_test_data = {
        "feature_rows": [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ],
        "target_values": [1.1, 2.2],
    }
    left_artifact = _build_artifact(ml_service)
    left_artifact["held_out_test_data"] = copy.deepcopy(held_out_test_data)
    left = ml_service.prepare_model_compare_artifact(
        left_artifact, filename="original.json"
    )
    right_artifact = _attach_edits_export(_build_artifact(ml_service), edits_payload)
    right_artifact["held_out_test_data"] = held_out_test_data
    right = ml_service.prepare_model_compare_artifact(
        right_artifact, filename="edited.json"
    )
    left["base_prediction_offset"] = 5.0
    right["base_prediction_offset"] = 5.0
    original_right = copy.deepcopy(right)
    options = dict(
        left_artifact=left,
        right_artifact=right,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        limit_edited_mean_deviation=True,
        manual_feature_shifts={"age": 10.0},
    )
    baseline = ml_service.build_model_compare_preview(**options)
    try:
        adjusted = ml_service.build_model_compare_preview(
            **options, intercept_adjustment=-36.0
        )
    except TypeError as exc:
        pytest.fail(f"Preview must accept an intercept adjustment: {exc}")

    assert baseline["mean_deviation_constraint"]["applied"] is True
    assert adjusted["mean_deviation_constraint"] == baseline["mean_deviation_constraint"]
    assert right == original_right
    assert adjusted["original_edited_base_prediction_offset"] == pytest.approx(5.0)
    assert adjusted["adjusted_edited_base_prediction_offset"] == pytest.approx(-31.0)
    assert adjusted["intercept_adjustment"] == pytest.approx(-36.0)
    for key in ("average_prediction_scores", "test_set_average_prediction_scores"):
        assert adjusted[key]["original"] == pytest.approx(baseline[key]["original"])
        assert adjusted[key]["reduced"] == pytest.approx(baseline[key]["reduced"])
        assert adjusted[key]["edited"] - baseline[key]["edited"] == pytest.approx(-36.0)
        assert adjusted[key]["difference"] - baseline[key]["difference"] == pytest.approx(-36.0)
        assert adjusted[key]["edited_minus_reduced"] - baseline[key]["edited_minus_reduced"] == pytest.approx(-36.0)
    for method in ("uniform", "test_set"):
        before = baseline["average_prediction_feature_contributions"][method]
        after = adjusted["average_prediction_feature_contributions"][method]
        for old_row, new_row in zip(before, after):
            delta = -36.0 if old_row["is_base_offset"] else 0.0
            assert new_row["edited"] - old_row["edited"] == pytest.approx(delta)
            assert new_row["original"] == pytest.approx(old_row["original"])
            assert new_row["reduced"] == pytest.approx(old_row["reduced"])
            assert new_row["difference"] - old_row["difference"] == pytest.approx(delta)
    assert adjusted["feature_previews"] == baseline["feature_previews"]
    before_test = baseline["held_out_test_performance"]
    after_test = adjusted["held_out_test_performance"]
    assert after_test["baseline"] == before_test["baseline"]
    assert after_test["baseline_predictions"] == before_test["baseline_predictions"]
    assert after_test["actual_values"] == before_test["actual_values"]
    assert [new - old for old, new in zip(
        before_test["edited_predictions"], after_test["edited_predictions"]
    )] == pytest.approx([-36.0, -36.0])
    errors = [prediction - actual for actual, prediction in zip(
        after_test["actual_values"], after_test["edited_predictions"]
    )]
    assert after_test["edited"]["mae"] == pytest.approx(
        sum(abs(error) for error in errors) / len(errors)
    )
    assert after_test["edited"]["rmse"] == pytest.approx(
        (sum(error ** 2 for error in errors) / len(errors)) ** 0.5
    )
    assert after_test["edited"]["mean_error"] == pytest.approx(
        sum(errors) / len(errors)
    )
    assert after_test["edited"]["r2"] == pytest.approx(
        1 - sum(error ** 2 for error in errors) /
        sum((actual - sum(after_test["actual_values"]) / len(errors)) ** 2
            for actual in after_test["actual_values"])
    )
    request = ModelComparePreviewRequest(
        left_artifact=left, right_artifact=right, intercept_adjustment=-36.0
    )
    response = ModelComparePreviewResponse(**adjusted)
    assert request.intercept_adjustment == -36.0
    assert response.adjusted_edited_base_prediction_offset == pytest.approx(-31.0)
    assert response.held_out_test_performance.edited.mean_error == pytest.approx(
        sum(errors) / len(errors)
    )
    assert response.held_out_test_performance.edited.r2 == pytest.approx(
        after_test["edited"]["r2"]
    )
    with pytest.raises(ValidationError):
        ModelComparePreviewRequest(
            left_artifact=left, right_artifact=right, intercept_adjustment=float("inf")
        )
    increased = ml_service.build_model_compare_preview(
        **options, intercept_adjustment=36.0
    )
    assert [new - old for old, new in zip(
        before_test["edited_predictions"],
        increased["held_out_test_performance"]["edited_predictions"],
    )] == pytest.approx([36.0, 36.0])


@pytest.mark.parametrize(
    ("test_source", "expected_targets", "original_mae", "edited_mae"),
    [
        ("left", [3.1, 4.2], 0.0, 1.0),
        ("right", [2.1, 3.2], 1.0, 0.0),
    ],
)
def test_held_out_performance_compares_original_and_edited_artifacts_on_selected_test_set(
    ml_service,
    test_source,
    expected_targets,
    original_mae,
    edited_mae,
):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-age",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 1.0},
                            {"x_value": "x:20", "y_value": 1.0, "weight": 1.0},
                        ],
                    }
                ],
            }
        ],
    }
    full_artifact = _build_artifact(ml_service)
    full_artifact["base_prediction_offset"] = 2.0
    full_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ],
        "target_values": [3.1, 4.2],
    }
    reduced_artifact = _attach_edits_export(
        _build_artifact(ml_service),
        edits_payload,
    )
    reduced_artifact["held_out_test_data"] = {
        "feature_rows": [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ],
        "target_values": [2.1, 3.2],
    }
    full = ml_service.prepare_model_compare_artifact(
        full_artifact,
        filename="main-page-model.json",
    )
    reduced = ml_service.prepare_model_compare_artifact(
        reduced_artifact,
        filename="reduced-data-model.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=full,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        feature_names=["age", "month"],
        held_out_test_source=test_source,
    )

    held_out = preview["held_out_test_performance"]
    assert held_out["test_data_source"] == test_source
    assert held_out["test_size"] == 2
    assert held_out["actual_values"] == pytest.approx(expected_targets)
    assert held_out["baseline_predictions"] == pytest.approx([3.1, 4.2])
    assert held_out["edited_predictions"] == pytest.approx([2.1, 3.2])
    assert held_out["baseline"]["mae"] == pytest.approx(original_mae)
    assert held_out["edited"]["mae"] == pytest.approx(edited_mae)
    assert held_out["edited"]["mae"] - held_out["baseline"]["mae"] == pytest.approx(
        edited_mae - original_mae
    )


def test_debug_performance_uses_main_page_training_rows_only(ml_service):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    ml_service.X_train = pd.DataFrame(
        {"age": [10.0, 20.0], "month": ["1", "2"]}
    )
    ml_service.y_train = pd.DataFrame({"target": [1.1, 2.2]})
    ml_service.X_test = pd.DataFrame({"age": [0.0], "month": ["3"]})
    ml_service.y_test = pd.DataFrame({"target": [-100.0]})

    options = dict(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        manual_feature_shifts={"age": 1.0},
    )
    prepared_preview = ml_service.build_model_compare_preview(**options)
    assert prepared_preview["main_training_performance_available"] is True
    preview = ml_service.build_model_compare_preview(
        **options, held_out_test_source="main_training"
    )
    assert preview["main_training_performance_available"] is True
    assert ModelComparePreviewResponse(**preview).main_training_performance_available is True
    performance = preview["held_out_test_performance"]
    assert performance["test_data_source"] == "main_training"
    assert performance["test_size"] == 2
    assert performance["actual_values"] == pytest.approx([1.1, 2.2])
    assert performance["baseline_predictions"] == pytest.approx([1.1, 2.2])
    assert performance["edited_predictions"] == pytest.approx([2.1, 3.2])
    assert performance["baseline"]["mae"] == pytest.approx(0.0)
    assert performance["edited"]["mae"] == pytest.approx(1.0)


def test_debug_performance_is_unavailable_without_compatible_main_training_data(
    ml_service,
):
    original = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service), filename="edited.json"
    )
    ml_service.y_train = pd.DataFrame({"target": [1.1]})
    ml_service.is_trained = False
    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        held_out_test_source="main_training",
    )
    assert preview["main_training_performance_available"] is False
    assert preview["held_out_test_performance"] is None

    ml_service.is_trained = True
    ml_service.X_train = pd.DataFrame({"age": [10.0]})
    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        held_out_test_source="main_training",
    )
    assert preview["main_training_performance_available"] is False
    assert preview["held_out_test_performance"] is None


def test_held_out_performance_uses_the_manual_feature_shift_shown_on_the_curve(
    ml_service,
):
    full_artifact = _build_artifact(ml_service)
    full_artifact["held_out_test_data"] = {
        "feature_rows": [{"age": 10.0, "month": "1"}],
        "target_values": [1.1],
    }
    full = ml_service.prepare_model_compare_artifact(
        full_artifact,
        filename="main-page-model.json",
    )
    reduced = ml_service.prepare_model_compare_artifact(
        _build_artifact(ml_service),
        filename="reduced-data-model.json",
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=full,
        right_artifact=reduced,
        left_selected_submission_ids=[],
        right_selected_submission_ids=[],
        feature_names=["age", "month"],
        manual_feature_shifts={"age": 2.0},
        held_out_test_source="left",
    )

    age_preview = _get_feature_preview(preview, "age")
    assert age_preview["right_effective_y_values"] == pytest.approx(
        [2.0, 3.0, 4.0, 5.0]
    )
    held_out = preview["held_out_test_performance"]
    assert held_out["baseline"]["mae"] == pytest.approx(0.0)
    assert held_out["edited"]["mae"] == pytest.approx(2.0)
    assert held_out["baseline"]["r2"] is None
    assert held_out["edited"]["r2"] is None
    assert held_out["baseline"]["mean_error"] == pytest.approx(0.0)
    assert held_out["edited"]["mean_error"] == pytest.approx(2.0)


def test_held_out_performance_uses_equal_edit_weighting_when_confidence_is_off(
    ml_service,
):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}, {"name": "bob"}],
        "edits": [
            {
                "user_name": "alice",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-alice",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 1.0, "weight": 0.2}
                        ],
                    }
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "age",
                        "feature_type": "numeric",
                        "submission_id": "sub-bob",
                        "edited_points": [
                            {"x_value": "x:10", "y_value": 3.0, "weight": 0.8}
                        ],
                    }
                ],
            },
        ],
    }
    full_artifact = _build_artifact(ml_service)
    full_artifact["held_out_test_data"] = {
        "feature_rows": [{"age": 10.0, "month": "1"}],
        "target_values": [1.1],
    }
    full = ml_service.prepare_model_compare_artifact(
        full_artifact,
        filename="main-page-model.json",
    )
    reduced = ml_service.prepare_model_compare_artifact(
        _attach_edits_export(_build_artifact(ml_service), edits_payload),
        filename="reduced-data-model.json",
    )
    preview_kwargs = {
        "left_artifact": full,
        "right_artifact": reduced,
        "left_selected_submission_ids": [],
        "right_selected_submission_ids": ["sub-alice", "sub-bob"],
        "feature_names": ["age", "month"],
        "held_out_test_source": "left",
    }

    weighted_preview = ml_service.build_model_compare_preview(
        **preview_kwargs,
        use_confidence=True,
    )
    unweighted_preview = ml_service.build_model_compare_preview(
        **preview_kwargs,
        use_confidence=False,
    )

    assert weighted_preview["held_out_test_performance"]["edited"]["mae"] == pytest.approx(
        1.3
    )
    assert unweighted_preview["held_out_test_performance"]["edited"]["mae"] == pytest.approx(
        2.0
    )
    corrected_unweighted = ml_service.build_model_compare_preview(
        **preview_kwargs,
        use_confidence=False,
        intercept_adjustment=-1.0,
    )
    assert corrected_unweighted["held_out_test_performance"]["edited_predictions"] == (
        pytest.approx([2.1])
    )
    assert corrected_unweighted["held_out_test_performance"]["intercept_correction"]["shift"] == (
        pytest.approx(-1.0)
    )


def test_model_export_includes_the_preprocessed_held_out_test_set(ml_service):
    ml_service.X_test = pd.DataFrame(
        [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ]
    )
    ml_service.y_test = pd.DataFrame({"target": [1.1, 2.2]})

    artifact = ml_service.export_model_artifact()

    assert artifact["held_out_test_data"] == {
        "feature_rows": [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ],
        "target_values": [1.1, 2.2],
        "sample_ids": ["0", "1"],
    }


def test_model_export_includes_preprocessed_training_feature_rows(ml_service):
    ml_service.X_train = pd.DataFrame(
        [
            {"age": 10.0, "month": "1"},
            {"age": 20.0, "month": "2"},
        ]
    )

    artifact = ml_service.export_model_artifact()
    assert artifact["training_feature_rows"] == [
        {"age": 10.0, "month": "1"},
        {"age": 20.0, "month": "2"},
    ]
    prepared = ml_service.prepare_model_compare_artifact(
        artifact, filename="reduced.json"
    )
    assert prepared["training_feature_rows"] == artifact["training_feature_rows"]
    ml_service.import_model_artifact(artifact)
    assert ml_service.export_model_artifact()["training_feature_rows"] == artifact[
        "training_feature_rows"
    ]


def test_training_mean_summary_uses_reduced_training_rows_before_and_after_correction(
    ml_service,
):
    edits_payload = {
        "included": True,
        "users": [{"name": "alice"}],
        "edits": [{
            "user_name": "alice",
            "shape_functions": [{
                "feature_name": "age",
                "feature_type": "numeric",
                "submission_id": "sub-age",
                "edited_points": [
                    {"x_value": "x:10", "y_value": 2.0, "weight": 1.0},
                    {"x_value": "x:20", "y_value": 2.0, "weight": 1.0},
                ],
            }],
        }],
    }
    original_artifact = _build_artifact(ml_service)
    original_artifact["base_prediction_offset"] = 10.0
    edited_artifact = _attach_edits_export(
        _build_artifact(ml_service), edits_payload
    )
    edited_artifact["base_prediction_offset"] = 5.0
    edited_artifact["training_feature_rows"] = [
        {"age": 10.0, "month": "1"},
        {"age": 20.0, "month": "2"},
    ]
    edited_artifact["training_sample_ids"] = ["0", "1"]
    original = ml_service.prepare_model_compare_artifact(
        original_artifact, filename="original.json"
    )
    edited = ml_service.prepare_model_compare_artifact(
        edited_artifact, filename="edited.json"
    )

    preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
        intercept_adjustment=-2.0,
    )

    summary = preview["training_mean_prediction_summary"]
    assert summary["training_row_count"] == 2
    assert summary["reduced_model_mean"] == pytest.approx(6.65)
    assert summary["edited_before_correction_mean"] == pytest.approx(8.65)
    assert summary["difference_before_correction"] == pytest.approx(2.0)
    assert summary["edited_after_correction_mean"] == pytest.approx(6.65)
    assert summary["reduced_intercept"] == pytest.approx(5.0)
    assert summary["edited_intercept_before_correction"] == pytest.approx(5.0)
    assert summary["intercept_adjustment"] == pytest.approx(-2.0)
    assert summary["edited_intercept_after_correction"] == pytest.approx(3.0)
    assert summary["suggested_matching_adjustment"] == pytest.approx(-2.0)
    assert summary["suggested_matching_intercept"] == pytest.approx(3.0)
    assert ModelComparePreviewResponse(**preview).training_mean_prediction_summary.training_row_count == 2

    legacy_edited = copy.deepcopy(edited)
    legacy_edited.pop("training_feature_rows")
    legacy_preview = ml_service.build_model_compare_preview(
        left_artifact=original,
        right_artifact=legacy_edited,
        left_selected_submission_ids=[],
        right_selected_submission_ids=["sub-age"],
    )
    assert legacy_preview["training_mean_prediction_summary"] is None


def test_model_export_uses_feature_specific_legacy_grids(ml_service):
    social = "Student Experience – Social Integration"
    anxiety = "German – Anxiety"
    climate = "German Instruction – Supportive Learning Climate"
    pressure = "German – Perceived Performance Pressure"
    ml_service.model = SimpleNamespace(
        boosting_rates=[],
        linear_model=SimpleNamespace(intercept_=0.0),
        get_params=lambda: {},
        get_shape_functions_as_dict=lambda: {
            social: {
                "x": [1.000000067, 1.25, 2.0, 3.999999974],
                "y": [0.0, 1.0, 2.0, 4.0],
                "datatype": "numeric",
            },
            anxiety: {
                "x": [1.000000067, 1.5, 2.0, 3.999999974],
                "y": [0.0, 1.0, 2.0, 4.0],
                "datatype": "numeric",
            },
            climate: {
                "x": [1.000000067, 1.5, 2.0, 3.999999974],
                "y": [0.0, 1.0, 2.0, 4.0],
                "datatype": "numeric",
            },
            pressure: {
                "x": [1.000000067, 1.25, 2.0, 3.999999974],
                "y": [0.0, 1.0, 2.0, 4.0],
                "datatype": "numeric",
            },
            "age": {
                "x": [0.0, 10.0, 20.0, 29.0],
                "y": [0.0, 1.0, 2.0, 3.0],
                "datatype": "numeric",
            },
        },
    )
    ml_service.feature_names = [social, anxiety, climate, pressure, "age"]
    ml_service.selected_feature_columns = list(ml_service.feature_names)
    ml_service.num_features = list(ml_service.feature_names)
    ml_service.cat_features = []
    ml_service.feature_schema = [
        {
            "name": social,
            "feature_type": "numeric",
            "min_value": 1.0,
            "max_value": 4.0,
        },
        {
            "name": anxiety,
            "feature_type": "numeric",
            "min_value": 1.0,
            "max_value": 4.0,
        },
        {
            "name": climate,
            "feature_type": "numeric",
            "min_value": 1.0,
            "max_value": 4.0,
        },
        {
            "name": pressure,
            "feature_type": "numeric",
            "min_value": 1.0,
            "max_value": 4.0,
        },
        {
            "name": "age",
            "feature_type": "numeric",
            "min_value": 0.0,
            "max_value": 29.0,
        },
    ]
    ml_service.feature_schema_map = {
        item["name"]: item for item in ml_service.feature_schema
    }
    ml_service.df = pd.DataFrame(
        {
            social: [1.0, 2.0, 4.0],
            anxiety: [1.0, 2.0, 4.0],
            climate: [1.0, 2.0, 4.0],
            pressure: [1.0, 2.0, 4.0],
            "age": [0, 10, 29],
        }
    )
    ml_service.X_train = ml_service.df.copy()

    artifact = ml_service.export_model_artifact()
    shapes = {shape["feature_name"]: shape for shape in artifact["shape_functions"]}

    assert shapes[social]["x_values"] == pytest.approx(
        [
            1.0,
            1.25,
            4 / 3,
            1.5,
            5 / 3,
            1.75,
            2.0,
            2.25,
            7 / 3,
            2.5,
            8 / 3,
            2.75,
            3.0,
            3.25,
            10 / 3,
            3.5,
            11 / 3,
            3.75,
            4.0,
        ]
    )
    assert shapes[anxiety]["x_values"][:6] == pytest.approx(
        [1.0, 9 / 8, 8 / 7, 7 / 6, 1.25, 9 / 7]
    )
    assert len(shapes[anxiety]["x_values"]) == 60
    assert shapes[climate]["x_values"] == pytest.approx(
        [1.0 + index / 7 for index in range(22)]
    )
    assert shapes[pressure]["x_values"] == pytest.approx(
        [1.0 + index / 5 for index in range(16)]
    )
    assert shapes["age"]["x_values"] == [0.0, 10.0, 20.0, 29.0]


def test_prepare_model_compare_rejects_unaligned_categorical_x_values(ml_service):
    original_artifact = _build_artifact(ml_service, target_column="ridership")
    edited_artifact = _build_artifact(ml_service, target_column="ridership")
    original = ml_service.prepare_model_compare_artifact(
        original_artifact,
        filename="original.json",
    )
    edited = ml_service.prepare_model_compare_artifact(
        edited_artifact,
        filename="edited.json",
    )
    _set_prepared_shape_values(
        edited,
        "month",
        x_values=["1", "3"],
        y_values=[0.1, 0.3],
    )

    with pytest.raises(ValueError, match="categorical feature 'month'"):
        ml_service._validate_model_compare_pair(original, edited)
