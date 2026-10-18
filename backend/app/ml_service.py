"""Shared state and public entry point for ML operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .chart_service import ChartServiceMixin
from .dataset_provenance import DatasetProvenanceMixin
from .dataset_runtime import DatasetRuntimeMixin
from .interactive_edits import InteractiveEditsMixin
from .ml_shared import CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN, ImportedIGANNRuntime, _json_safe
from .model_artifacts import ModelArtifactsMixin
from .model_compare_metrics import ModelCompareMetricsMixin
from .model_compare_prepare import ModelComparePrepareMixin
from .model_compare_preview import ModelComparePreviewMixin
from .model_prediction import ModelPredictionMixin


class MLService(
    DatasetProvenanceMixin,
    ChartServiceMixin,
    ModelArtifactsMixin,
    ModelComparePrepareMixin,
    ModelCompareMetricsMixin,
    ModelComparePreviewMixin,
    DatasetRuntimeMixin,
    ModelPredictionMixin,
    InteractiveEditsMixin,
):
    """Own active model state; domain operations live in focused mixins."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.preprocessor = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        self.imported_training_feature_rows: Optional[List[Dict[str, Any]]] = None
        self.df: Optional[pd.DataFrame] = None

        self.is_trained = False
        self.feature_names: List[str] = []
        self.selected_feature_columns: List[str] = []
        self.cat_features: List[str] = []
        self.num_features: List[str] = []
        self.target_column: Optional[str] = None

        self.feature_schema: List[Dict[str, Any]] = []
        self.feature_schema_map: Dict[str, Dict[str, Any]] = {}
        self.missing_indicator_map: Dict[str, str] = {}
        self.numeric_missing_placeholder_values: Dict[str, float] = {}
        self.show_missing_bars = False
        self.show_competence_levels = False
        self.show_prediction_comparison_superadmin = False
        self.show_prediction_comparison_users = False
        self.model_source: str = "trained"
        self.imported_artifact_version: Optional[str] = None
        self.analytics_available = False
        self.primary_n_estimators: Optional[int] = None

        # Store original shape functions for interactive editing
        self.original_shape_functions: Dict[str, Dict[str, Any]] = {}
        self.shape_function_offsets: Dict[str, Dict[Any, float]] = {}

        # Comparison dataset/model runtime (superadmin-only analysis state)
        self.comparison_model: Optional[Any] = None
        self.comparison_preprocessor = None
        self.comparison_X_train: Optional[pd.DataFrame] = None
        self.comparison_X_test: Optional[pd.DataFrame] = None
        self.comparison_y_train: Optional[pd.DataFrame] = None
        self.comparison_y_test: Optional[pd.DataFrame] = None
        self.comparison_df: Optional[pd.DataFrame] = None
        self.comparison_is_trained = False
        self.comparison_feature_names: List[str] = []
        self.comparison_selected_feature_columns: List[str] = []
        self.comparison_cat_features: List[str] = []
        self.comparison_num_features: List[str] = []
        self.comparison_target_column: Optional[str] = None
        self.comparison_feature_schema: List[Dict[str, Any]] = []
        self.comparison_feature_schema_map: Dict[str, Dict[str, Any]] = {}
        self.comparison_missing_indicator_map: Dict[str, str] = {}
        self.comparison_numeric_missing_placeholder_values: Dict[str, float] = {}
        self.comparison_original_shape_functions: Dict[str, Dict[str, Any]] = {}
        self.comparison_shape_function_offsets: Dict[str, Dict[Any, float]] = {}
        self.comparison_dataset_id: Optional[str] = None
        self.comparison_dataset_path: Optional[str] = None
        self.comparison_dataset_name: Optional[str] = None

        # Dataset persistence
        backend_dir = Path(__file__).resolve().parent.parent
        self.data_store_dir = backend_dir / "data_store"
        self.datasets_dir = self.data_store_dir / "datasets"
        self.active_dataset_file = self.data_store_dir / "active_dataset.json"
        self.data_store_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.active_dataset_id: Optional[str] = None
        self.active_dataset_path: Optional[str] = None
        self.active_dataset_name: Optional[str] = None
        self.feature_chart_settings: Dict[str, Dict[str, Any]] = {}
        self._feature_dictionary_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._column_mapping_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        self._restore_active_dataset_metadata()
        self._auto_load_persisted_dataset()

    def _clear_comparison_state(self) -> None:
        self.comparison_model = None
        self.comparison_preprocessor = None
        self.comparison_X_train = None
        self.comparison_X_test = None
        self.comparison_y_train = None
        self.comparison_y_test = None
        self.comparison_df = None
        self.comparison_is_trained = False
        self.comparison_feature_names = []
        self.comparison_selected_feature_columns = []
        self.comparison_cat_features = []
        self.comparison_num_features = []
        self.comparison_target_column = None
        self.comparison_feature_schema = []
        self.comparison_feature_schema_map = {}
        self.comparison_missing_indicator_map = {}
        self.comparison_numeric_missing_placeholder_values = {}
        self.comparison_original_shape_functions = {}
        self.comparison_shape_function_offsets = {}
        self.comparison_dataset_id = None
        self.comparison_dataset_path = None
        self.comparison_dataset_name = None

    def _snapshot_runtime_state(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "df": self.df,
            "is_trained": self.is_trained,
            "feature_names": list(self.feature_names),
            "selected_feature_columns": list(self.selected_feature_columns),
            "cat_features": list(self.cat_features),
            "num_features": list(self.num_features),
            "target_column": self.target_column,
            "feature_schema": list(self.feature_schema),
            "feature_schema_map": dict(self.feature_schema_map),
            "missing_indicator_map": dict(self.missing_indicator_map),
            "numeric_missing_placeholder_values": dict(
                self.numeric_missing_placeholder_values
            ),
            "show_missing_bars": bool(self.show_missing_bars),
            "show_competence_levels": bool(self.show_competence_levels),
            "original_shape_functions": dict(self.original_shape_functions),
            "shape_function_offsets": {
                feature: dict(offsets)
                for feature, offsets in self.shape_function_offsets.items()
            },
            "model_source": self.model_source,
            "imported_artifact_version": self.imported_artifact_version,
            "analytics_available": self.analytics_available,
        }

    def _restore_runtime_state(self, snapshot: Dict[str, Any]) -> None:
        for key, value in snapshot.items():
            setattr(self, key, value)

    def _has_analytics_data(self) -> bool:
        return (
            self.is_trained
            and self.X_test is not None
            and self.y_test is not None
            and self.model is not None
        )

    def _shape_functions_available(self) -> bool:
        return self.is_trained and (
            bool(self.original_shape_functions) or self.model is not None
        )

    def _require_analytics_data(self) -> None:
        if self._has_analytics_data():
            return
        if self.model_source == "imported":
            raise ValueError(
                "Analytics are unavailable for imported models until a compatible dataset is loaded"
            )
        raise ValueError("Model analytics are not available yet")

    def _ensure_original_shape_functions(self) -> None:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        if self.original_shape_functions:
            return
        if self.model_source == "imported":
            raise ValueError("Imported model shape functions are unavailable")

        existing_offsets = {
            feature: dict(offsets)
            for feature, offsets in self.shape_function_offsets.items()
        }
        shape_functions = self.get_shape_functions()
        self.original_shape_functions = {
            shape_function["feature_name"]: shape_function
            for shape_function in shape_functions
        }
        self.shape_function_offsets = existing_offsets


ml_service = MLService()
