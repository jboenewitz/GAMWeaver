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
    assert preview["use_confidence"] is True
    assert len(preview["feature_previews"]) == 2


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
    assert age_preview["left_effective_y_values"][1] == pytest.approx(1.6)
    assert age_preview["right_effective_y_values"][1] == pytest.approx(1.0)


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
    assert age_preview["left_effective_y_values"][1] == pytest.approx(2.5)


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
    assert age_preview["left_effective_y_values"][1] == pytest.approx(1.8)


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
    assert month_preview["left_effective_y_values"][1] == pytest.approx(0.7)


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

    assert _get_feature_preview(preview, "age")["left_effective_y_values"][1] == pytest.approx(1.5)
    assert ml_service.shape_function_offsets == before_shape_offsets
    assert ml_service.selected_feature_columns == before_selected_features
    assert ml_service.model_source == before_model_source
