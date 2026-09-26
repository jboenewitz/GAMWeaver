import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.ml_service import MLService


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
            "original": 1.7,
            "edited": 1.7,
            "difference": 0.0,
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
    assert constraint["applied"] is True
    assert constraint["scale_factor"] == pytest.approx(0.17)
    assert constraint["satisfied"] is True
    assert preview["average_prediction_scores"]["original"] == pytest.approx(1.7)
    assert preview["average_prediction_scores"]["edited"] == pytest.approx(1.87)
    age_preview = _get_feature_preview(preview, "age")
    assert (
        age_preview["right_effective_y_values"][1]
        - age_preview["right_base_y_values"][1]
    ) == pytest.approx(0.17)
    assert age_preview["right_individual_edit_traces"][0]["y_values"][1] == pytest.approx(
        age_preview["right_effective_y_values"][1]
    )


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
    assert age_preview["comparison_metrics"]["edited_vs_original_base_r2"] == pytest.approx(0.2)
    assert age_preview["comparison_metrics"]["edited_vs_edited_base_r2_status"] == "decreased"
    assert age_preview["comparison_metrics"]["edited_vs_original_base_r2_status"] == "decreased"
    assert preview["average_prediction_scores"] == pytest.approx(
        {
            "original": 1.7,
            "edited": 2.7,
            "difference": 1.0,
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
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_mae"] == pytest.approx(0.5)
    assert age_preview["comparison_metrics"]["edited_vs_original_base_mae"] == pytest.approx(0.0)
    assert age_preview["comparison_metrics"]["edited_vs_original_base_status"] == "decreased"
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_rmse"] == pytest.approx(0.7071067811865476)
    assert age_preview["comparison_metrics"]["edited_vs_original_base_rmse"] == pytest.approx(0.0)
    assert age_preview["comparison_metrics"]["baseline_edited_vs_original_base_r2"] == pytest.approx(0.6)
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
    assert alice_metrics["edited_vs_original_base_r2"] == pytest.approx(0.2)
    assert bob_metrics["edited_vs_original_base_r2"] == pytest.approx(-6.2)
    assert aggregated_metrics["edited_vs_original_base_r2"] == pytest.approx(-2.2)


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
