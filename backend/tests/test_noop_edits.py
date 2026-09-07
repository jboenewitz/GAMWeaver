from copy import deepcopy
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.db_service import DatabaseService
from backend.app.ml_service import MLService
from backend.app.noop_edits import (
    NOOP_EDIT_MESSAGE,
    complete_shape_function_edits_export,
)


BASE_SHAPES = [
    {
        "feature_name": "age",
        "feature_type": "numeric",
        "x_values": [1.0, 2.5],
    },
    {
        "feature_name": "school",
        "feature_type": "categorical",
        "x_values": ["primary", "secondary"],
    },
]


def _payload():
    return {
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
                            {"x_value": "x:1", "y_value": 2.0, "weight": 0.8}
                        ],
                    }
                ],
            },
            {
                "user_name": "bob",
                "shape_functions": [
                    {
                        "feature_name": "school",
                        "feature_type": "categorical",
                        "submission_id": "bob-school",
                        "edited_points": [
                            {"x_value": "primary", "y_value": -1.0, "weight": 0.6}
                        ],
                    }
                ],
            },
        ],
        "user_count": 2,
        "submission_count": 2,
    }


def test_completion_adds_valid_noops_and_is_idempotent():
    source = _payload()
    original_submission = deepcopy(source["edits"][0]["shape_functions"][0])

    completed = complete_shape_function_edits_export(source, BASE_SHAPES)
    repeated = complete_shape_function_edits_export(completed, BASE_SHAPES)

    assert completed["submission_count"] == 4
    assert completed["user_count"] == 2
    assert completed["edits"][0]["shape_functions"][0] == original_submission
    assert repeated == completed

    alice_school = next(
        shape
        for shape in completed["edits"][0]["shape_functions"]
        if shape["feature_name"] == "school"
    )
    bob_age = next(
        shape
        for shape in completed["edits"][1]["shape_functions"]
        if shape["feature_name"] == "age"
    )
    assert alice_school["is_imputed_noop"] is True
    assert alice_school["message"] == NOOP_EDIT_MESSAGE
    assert [point["x_value"] for point in alice_school["edited_points"]] == [
        "primary",
        "secondary",
    ]
    assert [point["x_value"] for point in bob_age["edited_points"]] == [
        "x:1",
        "x:2.5",
    ]
    assert all(point["y_value"] == 0.0 for point in bob_age["edited_points"])
    assert all(point["weight"] == 0.5 for point in bob_age["edited_points"])


def test_aggregation_counts_an_absent_chart_as_a_zero_noop(monkeypatch):
    service = DatabaseService()
    rows = [
        {
            "user_id": 1,
            "feature_name": "age",
            "feature_type": "numeric",
            "is_imputed_noop": False,
            "rows": [SimpleNamespace(x_value="x:1", y_offset=4.0, weight=0.5)],
        },
        {
            "user_id": 2,
            "feature_name": "school",
            "feature_type": "categorical",
            "is_imputed_noop": False,
            "rows": [
                SimpleNamespace(x_value="primary", y_offset=1.0, weight=0.5)
            ],
        },
    ]

    monkeypatch.setattr(service, "get_db", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(service, "_get_effective_submissions", lambda _db: rows)

    assert service.get_combined_edits(weighted=True)["age"]["x:1"] == 1.0
    assert service.get_combined_edits(weighted=False)["age"]["x:1"] == 2.0


def test_import_validation_preserves_the_noop_marker():
    validated = MLService._validate_shape_function_edits_export(
        {
            "included": True,
            "users": [{"name": "alice"}],
            "edits": [
                {
                    "user_name": "alice",
                    "shape_functions": [
                        {
                            "feature_name": "age",
                            "feature_type": "numeric",
                            "is_imputed_noop": True,
                            "edited_points": [
                                {"x_value": "x:1", "y_value": 0.0, "weight": 0.5}
                            ],
                        }
                    ],
                }
            ],
        },
        feature_names=["age"],
    )

    assert validated["edits"][0]["shape_functions"][0]["is_imputed_noop"] is True
