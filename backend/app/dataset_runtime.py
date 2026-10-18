"""Active dataset lifecycle, comparison dataset, and model training."""

from __future__ import annotations

from igann import IGANN
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

from .data_processing import prepare_training_data


class DatasetRuntimeMixin:
    def _validate_dataset_compatibility(
        self,
        *,
        feature_names: List[str],
        feature_schema: List[Dict[str, Any]],
    ) -> None:
        if list(feature_names) != list(self.selected_feature_columns):
            raise ValueError(
                "Loaded dataset is incompatible with the imported model: selected feature names do not match"
            )

        incoming_schema_map = {
            str(item["name"]): item for item in feature_schema if isinstance(item, dict) and item.get("name")
        }
        for imported_feature in self.selected_feature_columns:
            imported_schema = self.feature_schema_map.get(imported_feature)
            incoming_schema = incoming_schema_map.get(imported_feature)
            if imported_schema is None or incoming_schema is None:
                raise ValueError(
                    f"Loaded dataset is incompatible with the imported model: missing schema for '{imported_feature}'"
                )
            if str(imported_schema.get("feature_type")) != str(incoming_schema.get("feature_type")):
                raise ValueError(
                    f"Loaded dataset is incompatible with the imported model: feature type mismatch for '{imported_feature}'"
                )
            if str(imported_schema.get("feature_type")) == "categorical":
                allowed = {
                    str(value)
                    for value in (imported_schema.get("categorical_options") or [])
                }
                incoming = {
                    str(value)
                    for value in (incoming_schema.get("categorical_options") or [])
                }
                if not incoming.issubset(allowed):
                    raise ValueError(
                        f"Loaded dataset is incompatible with the imported model: categorical values for '{imported_feature}' are not representable by the imported schema"
                    )

    def get_model_status(self) -> Dict[str, Any]:
        self.analytics_available = self._has_analytics_data()
        return {
            "is_trained": self.is_trained,
            "data_loaded": self.X_train is not None,
            "features": self.selected_feature_columns if self.selected_feature_columns else [],
            "feature_schema": [
                self._with_feature_display_name(feature)
                for feature in (self.feature_schema or [])
            ],
            "target_column": self.target_column,
            "selected_feature_columns": self.selected_feature_columns if self.selected_feature_columns else [],
            "dataset_id": self.active_dataset_id,
            "dataset_name": self.active_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None),
            "train_size": len(self.X_train) if self.X_train is not None else 0,
            "test_size": len(self.X_test) if self.X_test is not None else 0,
            "model_source": self.model_source,
            "analytics_available": self.analytics_available,
            "shape_functions_available": self._shape_functions_available(),
            "imported_artifact_version": self.imported_artifact_version,
            "comparison_available": self.comparison_is_trained,
            "comparison_data_loaded": self.comparison_X_train is not None,
            "comparison_is_trained": self.comparison_is_trained,
            "comparison_dataset_name": self.comparison_dataset_name
            or (
                Path(self.comparison_dataset_path).name
                if self.comparison_dataset_path
                else None
            ),
            "comparison_train_size": (
                len(self.comparison_X_train)
                if self.comparison_X_train is not None
                else 0
            ),
            "primary_n_estimators": self.primary_n_estimators,
            "show_missing_bars": bool(self.show_missing_bars),
            "show_competence_levels": bool(self.show_competence_levels),
            "show_prediction_comparison_superadmin": bool(
                self.show_prediction_comparison_superadmin
            ),
            "show_prediction_comparison_users": bool(
                self.show_prediction_comparison_users
            ),
        }

    def _reset_runtime_state(self) -> None:
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.is_trained = False
        self.feature_names = []
        self.selected_feature_columns = []
        self.cat_features = []
        self.num_features = []
        self.feature_schema = []
        self.feature_schema_map = {}
        self.missing_indicator_map = {}
        self.numeric_missing_placeholder_values = {}
        self.model_source = "trained"
        self.imported_artifact_version = None
        self.analytics_available = False
        self.primary_n_estimators = None
        self.original_shape_functions = {}
        self.shape_function_offsets = {}
        self.target_column = None
        self.active_dataset_id = None
        self.active_dataset_path = None
        self.active_dataset_name = None
        self.feature_chart_settings = {}
        self.show_prediction_comparison_superadmin = False
        self.show_prediction_comparison_users = False
        self._clear_comparison_state()

    def reset_all_data(self) -> Dict[str, int]:
        """
        Reset ML runtime state and remove persisted uploaded dataset artifacts.

        This keeps the built-in fallback dataset (bike.csv) untouched, but clears
        uploaded CSV files, active dataset metadata, and extracted feature state.
        """
        deleted_dataset_files = 0
        deleted_dataset_dirs = 0
        deleted_metadata_files = 0

        if self.datasets_dir.exists():
            for entry in self.datasets_dir.iterdir():
                if entry.is_file() or entry.is_symlink():
                    entry.unlink(missing_ok=True)
                    deleted_dataset_files += 1
                elif entry.is_dir():
                    shutil.rmtree(entry)
                    deleted_dataset_dirs += 1

        if self.active_dataset_file.exists():
            self.active_dataset_file.unlink(missing_ok=True)
            deleted_metadata_files += 1

        self._reset_runtime_state()
        self.data_store_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        return {
            "deleted_dataset_files": deleted_dataset_files,
            "deleted_dataset_dirs": deleted_dataset_dirs,
            "deleted_metadata_files": deleted_metadata_files,
        }

    def load_data(
        self,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load and prepare the active dataset."""
        previous_dataset_path = self.active_dataset_path
        previous_target = self.target_column
        previous_selected_features = list(self.selected_feature_columns)
        existing_model = self.model
        existing_shape_functions = dict(self.original_shape_functions)
        existing_model_source = self.model_source
        existing_artifact_version = self.imported_artifact_version
        existing_chart_settings = dict(self.feature_chart_settings)
        previous_runtime_state = {
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
            "model_source": self.model_source,
            "imported_artifact_version": self.imported_artifact_version,
            "analytics_available": self.analytics_available,
            "original_shape_functions": dict(self.original_shape_functions),
            "shape_function_offsets": {
                feature: dict(offsets)
                for feature, offsets in self.shape_function_offsets.items()
            },
            "active_dataset_id": self.active_dataset_id,
            "active_dataset_path": self.active_dataset_path,
            "active_dataset_name": self.active_dataset_name,
            "feature_chart_settings": dict(self.feature_chart_settings),
        }

        resolved_path, resolved_dataset_id = self._resolve_dataset_path(dataset_id)
        resolved_path_str = str(resolved_path)

        # Switch to a different dataset file: discard previous chart settings.
        if previous_dataset_path and previous_dataset_path != resolved_path_str:
            self.feature_chart_settings = {}
        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            df_loaded,
            resolved_target,
            selected_features,
            cat_features,
            num_features,
            feature_processing_metadata,
        ) = prepare_training_data(
            csv_path=str(resolved_path),
            target_column=target_column,
            feature_columns=feature_columns,
        )

        public_numeric_features = list(
            feature_processing_metadata.get("public_numeric_features", [])
        )
        missing_indicator_map = dict(
            feature_processing_metadata.get("missing_indicator_map", {})
        )
        numeric_missing_placeholder_values = {
            str(key): float(value)
            for key, value in (
                feature_processing_metadata.get(
                    "numeric_missing_placeholder_values", {}
                )
                or {}
            ).items()
        }
        public_frame = df_loaded.loc[:, list(selected_features)].copy()
        rebuilt_feature_schema = self._build_public_feature_schema(
            list(selected_features),
            public_numeric_features,
            public_frame,
        )
        effective_dataset_name = (
            Path(str(dataset_name)).name
            if dataset_name
            else (
                self.active_dataset_name
                or (Path(resolved_path_str).name if resolved_path_str else None)
            )
        )
        rebuilt_feature_schema = self._enrich_feature_schema_with_provenance(
            rebuilt_feature_schema,
            dataset_name=effective_dataset_name,
            dataset_path=resolved_path,
        )
        rebuilt_feature_schema_map = {item["name"]: item for item in rebuilt_feature_schema}

        if existing_model_source == "imported" and existing_model is not None and self.is_trained:
            try:
                self._validate_dataset_compatibility(
                    feature_names=list(selected_features),
                    feature_schema=rebuilt_feature_schema,
                )
            except Exception:
                for key, value in previous_runtime_state.items():
                    setattr(self, key, value)
                raise

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = preprocessor
        self.df = df_loaded
        self.feature_names = list(X_train.columns)
        self.selected_feature_columns = list(selected_features)
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.target_column = resolved_target

        if existing_model_source == "imported" and existing_model is not None and previous_runtime_state["is_trained"]:
            self.model = existing_model
            self.is_trained = True
            self.model_source = existing_model_source
            self.imported_artifact_version = existing_artifact_version
            self.analytics_available = True
            self.feature_schema = list(self.feature_schema)
            self.feature_schema_map = dict(self.feature_schema_map)
            self.original_shape_functions = existing_shape_functions
            self.shape_function_offsets = {}
            self.feature_chart_settings = {
                feature: setting
                for feature, setting in existing_chart_settings.items()
                if feature in self.selected_feature_columns
            }
        else:
            # Any dataset load invalidates a trained local model and shape offsets.
            self.model = None
            self.is_trained = False
            self.model_source = "trained"
            self.imported_artifact_version = None
            self.analytics_available = False
            self.original_shape_functions = {}
            self.shape_function_offsets = {}
        self.feature_schema = rebuilt_feature_schema
        self.feature_schema_map = rebuilt_feature_schema_map
        self.missing_indicator_map = missing_indicator_map
        self.numeric_missing_placeholder_values = numeric_missing_placeholder_values
        self.feature_chart_settings = {
            feature: setting
            for feature, setting in self.feature_chart_settings.items()
            if feature in self.selected_feature_columns
        }

        self.active_dataset_id = resolved_dataset_id
        self.active_dataset_path = resolved_path_str
        self.active_dataset_name = (
            effective_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None)
        )
        self._persist_active_dataset_metadata()

        self._clear_comparison_state()

        dataset_changed = (
            previous_dataset_path != self.active_dataset_path
            or previous_target != self.target_column
            or previous_selected_features != self.selected_feature_columns
        )

        return {
            "total_records": len(self.df),
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "features": self.selected_feature_columns,
            "feature_schema": [
                self._with_feature_display_name(feature)
                for feature in self.feature_schema
            ],
            "target_column": self.target_column,
            "selected_feature_columns": self.selected_feature_columns,
            "dataset_id": self.active_dataset_id,
            "dataset_name": self.active_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None),
            "dataset_changed": dataset_changed,
        }

    def _validate_comparison_dataset_selection(
        self,
        *,
        target_column: str,
        feature_columns: List[str],
    ) -> None:
        if not self.is_trained or not self.primary_n_estimators:
            raise ValueError(
                "Train the primary model first so the comparison model can inherit its estimator count"
            )
        if target_column != self.target_column:
            raise ValueError(
                "Comparison dataset target column must match the primary dataset target column exactly"
            )
        if (
            len(feature_columns) != len(self.selected_feature_columns)
            or set(feature_columns) != set(self.selected_feature_columns)
        ):
            raise ValueError(
                "Comparison dataset feature columns must match the primary dataset selected feature columns exactly"
            )

    def _normalize_comparison_feature_order(
        self,
        feature_columns: List[str],
    ) -> List[str]:
        primary_feature_set = set(self.selected_feature_columns)
        normalized = [
            feature_name
            for feature_name in self.selected_feature_columns
            if feature_name in primary_feature_set and feature_name in set(feature_columns)
        ]
        if len(normalized) != len(feature_columns):
            return list(feature_columns)
        return normalized

    def _validate_comparison_feature_schema(
        self,
        feature_schema: List[Dict[str, Any]],
    ) -> None:
        """Validate comparison dataset feature compatibility without requiring identical derived defaults."""
        incoming_schema_map = {
            str(item["name"]): item
            for item in feature_schema
            if isinstance(item, dict) and item.get("name")
        }
        primary_schema_map = {
            str(item["name"]): item
            for item in self.feature_schema
            if isinstance(item, dict) and item.get("name")
        }

        for feature_name in self.selected_feature_columns:
            primary_schema = primary_schema_map.get(feature_name)
            comparison_schema = incoming_schema_map.get(feature_name)
            if primary_schema is None or comparison_schema is None:
                raise ValueError(
                    f"Comparison dataset schema is incompatible: missing schema for '{feature_name}'"
                )

            primary_type = str(primary_schema.get("feature_type"))
            comparison_type = str(comparison_schema.get("feature_type"))
            if primary_type != comparison_type:
                raise ValueError(
                    f"Comparison dataset schema is incompatible: feature type mismatch for '{feature_name}'"
                )

            if primary_type == "categorical":
                primary_options = {
                    str(value)
                    for value in (primary_schema.get("categorical_options") or [])
                }
                comparison_options = {
                    str(value)
                    for value in (comparison_schema.get("categorical_options") or [])
                }
                if not comparison_options.issubset(primary_options):
                    raise ValueError(
                        f"Comparison dataset schema is incompatible: categorical values for '{feature_name}' are not representable by the primary dataset schema"
                    )

    def load_comparison_data(
        self,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load and validate a second dataset for shape-function comparison."""
        resolved_path, resolved_dataset_id = self._resolve_dataset_path(dataset_id)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            df_loaded,
            resolved_target,
            selected_features,
            cat_features,
            num_features,
            feature_processing_metadata,
        ) = prepare_training_data(
            csv_path=str(resolved_path),
            target_column=target_column,
            feature_columns=feature_columns,
        )

        selected_features = list(selected_features)
        self._validate_comparison_dataset_selection(
            target_column=resolved_target,
            feature_columns=selected_features,
        )
        selected_features = self._normalize_comparison_feature_order(
            selected_features,
        )

        public_numeric_features = list(
            feature_processing_metadata.get("public_numeric_features", [])
        )
        public_numeric_features = [
            feature_name
            for feature_name in selected_features
            if feature_name in set(public_numeric_features)
        ]
        comparison_missing_indicator_map = dict(
            feature_processing_metadata.get("missing_indicator_map", {})
        )
        comparison_numeric_missing_placeholder_values = {
            str(key): float(value)
            for key, value in (
                feature_processing_metadata.get(
                    "numeric_missing_placeholder_values", {}
                )
                or {}
            ).items()
        }
        public_frame = df_loaded.loc[:, list(selected_features)].copy()
        rebuilt_feature_schema = self._build_public_feature_schema(
            list(selected_features),
            public_numeric_features,
            public_frame,
        )
        comparison_dataset_name = (
            Path(str(dataset_name)).name
            if dataset_name
            else Path(str(resolved_path)).name
        )
        rebuilt_feature_schema = self._enrich_feature_schema_with_provenance(
            rebuilt_feature_schema,
            dataset_name=comparison_dataset_name,
            dataset_path=resolved_path,
        )
        self._validate_comparison_feature_schema(rebuilt_feature_schema)

        self.comparison_model = None
        self.comparison_is_trained = False
        self.comparison_original_shape_functions = {}
        self.comparison_shape_function_offsets = {}
        self.comparison_X_train = X_train
        self.comparison_X_test = X_test
        self.comparison_y_train = y_train
        self.comparison_y_test = y_test
        self.comparison_preprocessor = preprocessor
        self.comparison_df = df_loaded
        self.comparison_feature_names = list(X_train.columns)
        self.comparison_selected_feature_columns = selected_features
        self.comparison_cat_features = list(cat_features)
        self.comparison_num_features = list(num_features)
        self.comparison_target_column = resolved_target
        self.comparison_feature_schema = rebuilt_feature_schema
        self.comparison_feature_schema_map = {
            item["name"]: item for item in rebuilt_feature_schema
        }
        self.comparison_missing_indicator_map = comparison_missing_indicator_map
        self.comparison_numeric_missing_placeholder_values = (
            comparison_numeric_missing_placeholder_values
        )
        self.comparison_dataset_id = resolved_dataset_id
        self.comparison_dataset_path = str(resolved_path)
        self.comparison_dataset_name = comparison_dataset_name

        return {
            "total_records": len(df_loaded),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": selected_features,
            "target_column": resolved_target,
            "selected_feature_columns": selected_features,
            "dataset_id": self.comparison_dataset_id,
            "dataset_name": self.comparison_dataset_name,
            "primary_n_estimators": self.primary_n_estimators,
        }

    def train_comparison_model(self) -> int:
        """Train the comparison model using the primary model's estimator count."""
        if not self.is_trained or not self.primary_n_estimators:
            raise ValueError(
                "Train the primary model first so the comparison model can inherit its estimator count"
            )
        if self.comparison_X_train is None or self.comparison_y_train is None:
            raise ValueError("Load a comparison dataset first")

        self.comparison_model = IGANN(
            task="regression",
            n_estimators=self.primary_n_estimators,
            verbose=0,
            scale_y=True,
        )
        self.comparison_model.fit(self.comparison_X_train, self.comparison_y_train)
        self.comparison_is_trained = True
        self.comparison_original_shape_functions = {}
        self.comparison_shape_function_offsets = {}
        return int(self.primary_n_estimators)
