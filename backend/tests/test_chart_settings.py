import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.ml_service import MLService
from backend.app.security import create_admin_token


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


def test_legacy_value_labels_migrate_to_categorical_labels(ml_service):
    normalized = ml_service._normalize_stored_feature_chart_setting(
        {
            "treat_as_categorical": True,
            "value_labels": {"1": "One", "2": "  "},
        }
    )

    assert normalized["treat_as_categorical"] is True
    assert normalized["categorical_value_labels"] == {"1": "One"}
    assert normalized["numeric_tick_labels"] == {}


def test_numeric_tick_labels_are_saved_and_returned_for_numeric_feature(ml_service):
    result = ml_service.update_feature_chart_setting(
        "age",
        treat_as_categorical=False,
        numeric_tick_labels={"0": "Start", "10": "Mid", "999": "Ignore"},
    )

    assert result["chart_feature_type"] == "numeric"
    assert result["numeric_tick_labels"] == {"0": "Start", "10": "Mid"}
    assert result["available_numeric_values"][:3] == ["0", "1", "2"]


def test_numeric_label_options_are_limited_to_whole_number_values(ml_service):
    scaled = [1 + (3 * idx / 29) for idx in range(30)]
    ml_service.df["age"] = scaled
    ml_service.X_train["age"] = scaled
    ml_service.feature_schema_map["age"]["min_value"] = 1.0
    ml_service.feature_schema_map["age"]["max_value"] = 4.0

    result = ml_service.get_feature_chart_setting("age")

    assert result["available_numeric_values"] == ["1", "2", "3", "4"]
    assert result["numeric_domain_min"] == 1.0
    assert result["numeric_domain_max"] == 4.0


def test_item_mean_provenance_overrides_numeric_domain_to_full_scale(ml_service):
    domain = ml_service._infer_numeric_domain_from_provenance(
        {"name": "age", "feature_type": "numeric"},
        {
            "construction_type": "item_mean",
            "response_options": [
                {"value": "1.0", "label": "stimmt gar nicht"},
                {"value": "2.0", "label": "stimmt eher nicht"},
                {"value": "3.0", "label": "stimmt eher"},
                {"value": "4.0", "label": "stimmt genau"},
            ],
        },
    )

    assert domain == (1.0, 4.0)


def test_switching_chart_modes_preserves_both_label_maps(ml_service):
    initial = ml_service.update_feature_chart_setting(
        "month",
        treat_as_categorical=False,
        treat_as_numeric=False,
        categorical_value_labels={"1": "January"},
        numeric_tick_labels={"1": "One"},
    )
    assert initial["chart_feature_type"] == "categorical"

    numeric = ml_service.update_feature_chart_setting(
        "month",
        treat_as_categorical=False,
        treat_as_numeric=True,
    )
    assert numeric["chart_feature_type"] == "numeric"
    assert numeric["categorical_value_labels"] == {"1": "January"}
    assert numeric["numeric_tick_labels"] == {"1": "One"}

    categorical = ml_service.update_feature_chart_setting(
        "month",
        treat_as_categorical=False,
        treat_as_numeric=False,
    )
    assert categorical["chart_feature_type"] == "categorical"
    assert categorical["categorical_value_labels"] == {"1": "January"}
    assert categorical["numeric_tick_labels"] == {"1": "One"}


def test_numeric_shape_functions_include_sparse_tick_labels(ml_service):
    ml_service.update_feature_chart_setting(
        "age",
        treat_as_categorical=False,
        numeric_tick_labels={"0": "Start", "10": "Checkpoint"},
    )

    shape_function = ml_service._extract_shape_function("age", 0)

    assert shape_function["feature_type"] == "numeric"
    assert len(shape_function["x_tick_labels"]) == len(shape_function["x_values"])
    label_by_x = {
        int(x_value): label
        for x_value, label in zip(
            shape_function["x_values"], shape_function["x_tick_labels"]
        )
        if label
    }
    assert label_by_x == {0: "Start", 10: "Checkpoint"}


def test_export_import_roundtrip_preserves_chart_label_maps(ml_service, monkeypatch):
    ml_service.update_feature_chart_setting(
        "age",
        treat_as_categorical=False,
        numeric_tick_labels={"0": "Start", "20": "Twenty"},
    )
    ml_service.update_feature_chart_setting(
        "month",
        treat_as_categorical=False,
        treat_as_numeric=False,
        categorical_value_labels={"1": "January"},
    )

    artifact = ml_service.export_model_artifact()

    monkeypatch.setattr(MLService, "_restore_active_dataset_metadata", lambda self: None)
    monkeypatch.setattr(MLService, "_auto_load_persisted_dataset", lambda self: None)
    imported = MLService()
    imported.import_model_artifact(artifact)

    assert imported.feature_chart_settings["age"]["numeric_tick_labels"] == {
        "0": "Start",
        "20": "Twenty",
    }
    assert imported.feature_chart_settings["month"][
        "categorical_value_labels"
    ] == {"1": "January"}


def test_feature_chart_settings_require_superadmin(ml_service, monkeypatch):
    from backend.app import main as main_module

    monkeypatch.setattr(main_module, "ml_service", ml_service)
    monkeypatch.setattr(main_module.db_service, "ensure_superadmin", lambda: None)

    request = Request({"type": "http", "headers": []})

    with pytest.raises(HTTPException) as exc_info:
        anyio.run(
            main_module.update_feature_chart_settings,
            "age",
            main_module.FeatureChartSettingsRequest(
                numeric_tick_labels={"0": "Start"},
            ),
            request,
        )

    assert exc_info.value.status_code == 403


def test_superadmin_can_still_use_legacy_value_labels_request(ml_service, monkeypatch):
    from backend.app import main as main_module

    monkeypatch.setattr(main_module, "ml_service", ml_service)
    monkeypatch.setattr(main_module.db_service, "ensure_superadmin", lambda: None)
    monkeypatch.setattr(main_module.settings, "auth_token_secret", "test-secret")
    monkeypatch.setattr(main_module.settings, "superadmin_username", "superadmin")
    monkeypatch.setattr(main_module.settings, "admin_token_ttl_hours", 1)

    token = create_admin_token("superadmin", "test-secret", 3600)
    request = Request(
        {
            "type": "http",
            "headers": [(b"x-superadmin-token", token.encode("utf-8"))],
        }
    )

    response = anyio.run(
        main_module.update_feature_chart_settings,
        "month",
        main_module.FeatureChartSettingsRequest(
            value_labels={"1": "January"},
            treat_as_numeric=False,
        ),
        request,
    )

    assert response.categorical_value_labels == {"1": "January"}
    assert response.value_labels == {"1": "January"}


def test_comparison_feature_validation_allows_reordered_columns(ml_service):
    ml_service.primary_n_estimators = 10

    ml_service._validate_comparison_dataset_selection(
        target_column=ml_service.target_column,
        feature_columns=["month", "age"],
    )

    normalized = ml_service._normalize_comparison_feature_order(
        ["month", "age"],
    )
    assert normalized == ["age", "month"]


def test_load_comparison_data_keeps_internal_missing_indicator_features(
    ml_service,
    monkeypatch,
):
    ml_service.primary_n_estimators = 10
    ml_service.target_column = "target"
    ml_service.selected_feature_columns = ["age", "month"]
    ml_service.feature_schema = [
        {"name": "age", "feature_type": "numeric"},
        {"name": "month", "feature_type": "categorical", "categorical_options": ["1", "2"]},
    ]

    X_train = pd.DataFrame(
        {
            "age": [1.0, 2.0],
            "age [missing]": ["Observed", "Observed"],
            "month": ["1", "2"],
        }
    )
    X_test = X_train.copy()
    y_train = pd.DataFrame({"target": [10.0, 20.0]})
    y_test = y_train.copy()
    df_loaded = pd.DataFrame({"age": [1.0, 2.0], "month": ["1", "2"], "target": [10.0, 20.0]})

    monkeypatch.setattr(
        "backend.app.ml_service.prepare_training_data",
        lambda **kwargs: (
            X_train,
            X_test,
            y_train,
            y_test,
            object(),
            df_loaded,
            "target",
            ["month", "age"],
            ["month", "age [missing]"],
            ["age"],
            {
                "public_numeric_features": ["age"],
                "missing_indicator_map": {"age": "age [missing]"},
                "numeric_missing_placeholder_values": {"age": 1.5},
            },
        ),
    )
    monkeypatch.setattr(
        ml_service,
        "_resolve_dataset_path",
        lambda dataset_id=None: (Path("/tmp/comparison.csv"), "comparison.csv"),
    )
    monkeypatch.setattr(
        ml_service,
        "_enrich_feature_schema_with_provenance",
        lambda feature_schema, **kwargs: feature_schema,
    )

    ml_service.load_comparison_data(
        dataset_id="comparison.csv",
        target_column="target",
        feature_columns=["month", "age"],
    )

    assert ml_service.comparison_selected_feature_columns == ["age", "month"]
    assert ml_service.comparison_cat_features == ["month", "age [missing]"]
    assert ml_service.comparison_num_features == ["age"]
