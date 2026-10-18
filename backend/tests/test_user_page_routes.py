"""HTTP behavior used by the main page and combined results page."""

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import user_routes
from backend.app.main import app


async def request(method, path, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_main_page_status_shapes_and_prediction(monkeypatch):
    seen_features = []
    shape = {
        "feature_name": "age",
        "feature_type": "numeric",
        "x_values": [0.0, 10.0],
        "y_values": [1.0, 2.0],
    }
    service = SimpleNamespace(
        is_trained=True,
        get_model_status=lambda: {
            "is_trained": True,
            "data_loaded": True,
            "features": ["age"],
            "feature_schema": [{"name": "age", "feature_type": "numeric"}],
            "selected_feature_columns": ["age"],
            "target_column": "score",
            "shape_functions_available": True,
        },
        get_shape_functions=lambda: [shape],
        get_comparison_shape_functions=lambda: [],
        predict=lambda features: seen_features.append(features) or -2.0,
    )
    monkeypatch.setattr(user_routes, "ml_service", service)

    assert (await request("GET", "/api/model/status")).json()["features"] == ["age"]
    assert (await request("GET", "/api/model/shape-functions")).json()["shape_functions"] == [shape | {
        "x_tick_labels": None,
        "chart_config": None,
        "distribution": None,
        "missing_bucket": None,
    }]
    prediction = await request("POST", "/api/predict", json={"features": {"age": 4}})
    assert prediction.status_code == 200
    assert prediction.json() == {"predicted_count": 0.0, "input_features": {"age": 4}}
    assert seen_features == [{"age": 4}]
    assert (await request("POST", "/api/predict", json={"features": []})).status_code == 400


@pytest.mark.asyncio
async def test_combined_results_use_requested_weighting_and_restore_user_offsets(monkeypatch):
    class Service:
        is_trained = True
        model_source = "trained"
        original_shape_functions = {
            "age": {
                "feature_name": "age",
                "feature_type": "numeric",
                "x_values": [0.0, 10.0],
                "y_values": [1.0, 2.0],
            }
        }

        def __init__(self):
            self.shape_function_offsets = {"age": {0.0: 9.0}}
            self.seen_offsets = []

        def get_model_status(self):
            return {"analytics_available": True}

        def get_predictions_comparison(self):
            self.seen_offsets.append(self.shape_function_offsets.copy())
            return {"original_predictions": [1.0], "interactive_predictions": [2.0], "actual_values": [1.5]}

        def get_comparison_shape_functions(self):
            return []

        def get_offset_for_feature_value(self, *, value, offsets, **kwargs):
            return offsets.get(value, 0.0)

    service = Service()
    requested_weighting = []

    class Database:
        def get_combined_edits(self, weighted=True):
            requested_weighting.append(weighted)
            return {"age": {0.0: 0.5 if weighted else 1.0}}

        def get_combined_edits_detailed(self):
            return {"total_users_with_edits": 1, "shape_functions": []}

        def get_users_with_edits(self):
            return [{"id": 7, "name": "expert"}]

        def get_user_edits_raw(self, user_id):
            assert user_id == 7
            return {"age": {0.0: {"offset": 2.0, "weight": 0.25}}}

    monkeypatch.setattr(user_routes, "ml_service", service)
    monkeypatch.setattr(user_routes, "db_service", Database())

    weighted = await request("GET", "/api/combined/predictions-comparison?weighted=true")
    plain = await request("GET", "/api/combined/predictions-comparison?weighted=false")
    assert weighted.status_code == plain.status_code == 200
    assert requested_weighting == [True, False]
    assert service.seen_offsets == [{"age": {0.0: 0.5}}, {"age": {0.0: 1.0}}]
    assert service.shape_function_offsets == {"age": {0.0: 9.0}}
    assert weighted.json()["combined_shape_functions_display"][0]["y_values"] == [1.5, 2.0]
    assert plain.json()["combined_shape_functions_display"][0]["y_values"] == [2.0, 2.0]

    overlay = await request("GET", "/api/combined/per-user-shape-functions?weighted=true")
    assert overlay.status_code == 200
    assert overlay.json()["users"][0]["shape_functions"][0]["y_values"] == [1.5, 2.0]


@pytest.mark.asyncio
async def test_combined_prediction_uses_aggregate_offsets(monkeypatch):
    received = []
    service = SimpleNamespace(
        is_trained=True,
        original_shape_functions={"age": {}},
        predict_with_offsets=lambda features, offsets: received.append((features, offsets)) or 3.5,
    )
    database = SimpleNamespace(get_combined_edits=lambda: {"age": {0.0: 0.5}})
    monkeypatch.setattr(user_routes, "ml_service", service)
    monkeypatch.setattr(user_routes, "db_service", database)

    response = await request("POST", "/api/combined/predict", json={"features": {"age": 4}})
    assert response.status_code == 200
    assert response.json() == {"predicted_count": 3.5, "input_features": {"age": 4}}
    assert received == [({"age": 4}, {"age": {0.0: 0.5}})]


@pytest.mark.asyncio
async def test_main_page_edit_submission_is_saved_for_combined_results(monkeypatch):
    submitted = []
    saved = []

    service = SimpleNamespace(
        is_trained=True,
        update_shape_functions=lambda edits: submitted.extend(edits),
        convert_edits_for_storage=lambda edits: [
            {"feature_name": edits[0]["feature_name"], "edited_points": edits[0]["edited_points"]}
        ],
    )
    database = SimpleNamespace(
        get_user_by_id=lambda user_id: {"id": user_id, "name": "expert"},
        save_user_edits=lambda user_id, edits: saved.append((user_id, edits)),
        get_users_with_edits=lambda: [{"id": 7, "name": "expert"}],
        get_edit_logs=lambda: {"age": [{"user_name": "expert"}]},
    )
    monkeypatch.setattr(user_routes, "ml_service", service)
    monkeypatch.setattr(user_routes, "db_service", database)

    response = await request(
        "POST",
        "/api/users/7/edits",
        json={"edited_shape_functions": [{
            "feature_name": "age",
            "feature_type": "numeric",
            "edited_points": [{
                "x_value": 10,
                "y_value": 2.5,
                "weight": 0.8,
                "message": "Reviewed",
            }],
        }]},
    )
    assert response.status_code == 200
    assert submitted[0]["edited_points"] == [{
        "x_value": 10,
        "y_value": 2.5,
        "weight": 0.8,
        "message": "Reviewed",
    }]
    assert saved[0][0] == 7
    assert len(saved[0][1][0]["submission_id"]) == 32
    assert (await request("GET", "/api/combined/users")).json() == {
        "users": [{"id": 7, "name": "expert"}]
    }
    assert (await request("GET", "/api/combined/edit-logs")).json() == {
        "age": [{"user_name": "expert"}]
    }
