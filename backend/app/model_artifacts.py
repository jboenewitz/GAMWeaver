"""Export and import model artifacts, including validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data_processing import MISSING_CATEGORY_VALUE, MISSING_INDICATOR_SUFFIX
from .ml_shared import (
    DEFAULT_LEGACY_ONE_TO_FOUR_EXPORT_GRID,
    ImportedIGANNRuntime,
    LEGACY_NUMERIC_EXPORT_GRIDS,
    MODEL_ARTIFACT_VERSION,
    MODEL_ARTIFACT_VERSION_WITH_NOOP_EDITS,
    SUPPORTED_MODEL_ARTIFACT_VERSIONS,
    _json_safe,
)


class ModelArtifactsMixin:
    def _get_legacy_compatible_export_shape_values(
        self,
        feature_name: str,
        x_values: List[Any],
        y_values: List[Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Use historic export grids for numeric 1--4 questionnaire scales.

        Older artifacts used a measurement-specific grid for each known scale.
        Keeping those grids makes new exports directly comparable while leaving
        other numeric domains (for example age or ISEI) untouched. Unknown 1--4
        scales use the historic 16-point default grid.
        """
        schema = self.feature_schema_map.get(feature_name, {}) or {}
        try:
            schema_min = float(schema.get("min_value"))
            schema_max = float(schema.get("max_value"))
        except (TypeError, ValueError):
            return x_values, y_values

        if not (
            np.isclose(schema_min, 1.0)
            and np.isclose(schema_max, 4.0)
            and len(x_values) == len(y_values)
        ):
            return x_values, y_values

        try:
            source_x = np.asarray(x_values, dtype=float)
            source_y = np.asarray(y_values, dtype=float)
        except (TypeError, ValueError):
            return x_values, y_values

        valid = np.isfinite(source_x) & np.isfinite(source_y)
        source_x = source_x[valid]
        source_y = source_y[valid]
        if source_x.size < 2:
            return x_values, y_values

        sort_order = np.argsort(source_x)
        source_x = source_x[sort_order]
        source_y = source_y[sort_order]
        unique_x, unique_indices = np.unique(source_x, return_index=True)
        source_y = source_y[unique_indices]
        endpoint_tolerance = 1e-6
        if (
            unique_x.size < 2
            or unique_x[0] > 1.0 + endpoint_tolerance
            or unique_x[-1] < 4.0 - endpoint_tolerance
        ):
            return x_values, y_values

        legacy_grid = np.asarray(
            LEGACY_NUMERIC_EXPORT_GRIDS.get(
                feature_name,
                DEFAULT_LEGACY_ONE_TO_FOUR_EXPORT_GRID,
            ),
            dtype=float,
        )
        return legacy_grid.tolist(), np.interp(legacy_grid, unique_x, source_y).tolist()

    def _export_shape_functions_payload(self) -> List[Dict[str, Any]]:
        if self.model_source == "imported":
            if isinstance(self.model, ImportedIGANNRuntime):
                return self.model.export_shape_functions()
            self._ensure_original_shape_functions()
            return [
                _json_safe(shape_function)
                for shape_function in self.original_shape_functions.values()
            ]

        if self.model is None or not hasattr(self.model, "get_shape_functions_as_dict"):
            raise ValueError("Unable to export complete shape functions for the trained model")

        raw_shape_functions = self.model.get_shape_functions_as_dict()
        exported: List[Dict[str, Any]] = []
        for feature_name in self.feature_names:
            raw_shape = raw_shape_functions.get(feature_name)
            if raw_shape is None:
                continue
            chart_config = {}
            if feature_name in self.feature_names:
                try:
                    chart_config = self.get_feature_chart_setting(feature_name)
                except Exception:
                    chart_config = {}
            feature_type = (
                "categorical"
                if str(raw_shape.get("datatype", "")).strip() == "categorical"
                else "numeric"
            )
            exported_shape = {
                "feature_name": feature_name,
                "x_values": _json_safe(raw_shape.get("x", [])),
                "y_values": _json_safe(raw_shape.get("y", [])),
                "feature_type": feature_type,
                "chart_config": chart_config,
            }
            if feature_type == "numeric":
                (
                    exported_shape["x_values"],
                    exported_shape["y_values"],
                ) = self._get_legacy_compatible_export_shape_values(
                    feature_name,
                    exported_shape["x_values"],
                    exported_shape["y_values"],
                )
            if feature_type == "categorical":
                exported_shape["x_tick_labels"] = [
                    str(value) for value in _json_safe(raw_shape.get("x", []))
                ]
            distribution = self._build_shape_function_distribution(
                feature_name,
                feature_type,
                exported_shape["x_values"],
                x_tick_labels=exported_shape.get("x_tick_labels"),
            )
            if distribution is not None:
                exported_shape["distribution"] = _json_safe(distribution)
            exported.append(exported_shape)
        if not exported:
            raise ValueError("Unable to export any shape functions for the trained model")
        return exported

    def _infer_base_prediction_offset(self) -> float:
        if self.model is None:
            raise ValueError("Model not trained yet")
        if self.model_source == "imported":
            return float(getattr(self.model, "base_prediction_offset", 0.0))

        y_scaler = getattr(self.model, "y_scaler", None)
        mean_values = getattr(y_scaler, "mean_", None)
        if mean_values is not None and len(mean_values):
            return float(mean_values[0])
        intercept = getattr(getattr(self.model, "linear_model", None), "intercept_", 0.0)
        if isinstance(intercept, np.ndarray):
            if intercept.size:
                return float(intercept.reshape(-1)[0])
            return 0.0
        return float(intercept)

    def _export_held_out_test_data(self) -> Optional[Dict[str, Any]]:
        """Serialize the exact preprocessed test split used during training."""
        if (
            self.X_test is None
            or self.y_test is None
            or not self.feature_names
            or self.X_test.empty
            or self.y_test.empty
        ):
            return None

        feature_rows = self.X_test.loc[:, self.feature_names].to_dict(
            orient="records"
        )
        target_values = self.y_test.iloc[:, 0].tolist()
        if len(feature_rows) != len(target_values):
            return None

        return {
            "feature_rows": _json_safe(feature_rows),
            "target_values": _json_safe(target_values),
            "sample_ids": [str(index) for index in self.X_test.index],
        }

    def _export_training_feature_rows(self) -> Optional[List[Dict[str, Any]]]:
        """Serialize preprocessed training rows for artifact-based mean predictions."""
        if (
            self.X_train is not None
            and not self.X_train.empty
            and self.feature_names
            and set(self.feature_names).issubset(self.X_train.columns)
        ):
            return _json_safe(
                self.X_train.loc[:, self.feature_names].to_dict(orient="records")
            )
        if self.model_source == "imported":
            return self.imported_training_feature_rows
        return None

    def export_model_artifact(
        self,
        *,
        include_shape_function_edits: bool = False,
    ) -> Dict[str, Any]:
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        igann_params = {}
        if hasattr(self.model, "get_params"):
            igann_params = _json_safe(self.model.get_params())

        shape_functions = [
            {
                **shape_function,
                "chart_config": {
                    key: value
                    for key, value in ((shape_function.get("chart_config") or {}).items())
                    if key not in {"display_title", "x_axis_label"}
                },
            }
            for shape_function in self._export_shape_functions_payload()
        ]
        exported_feature_names = [
            shape_function["feature_name"] for shape_function in shape_functions
        ]
        exported_feature_schema = [
            _json_safe(item)
            for item in self.feature_schema
            if str(item.get("name")) in exported_feature_names
        ]
        exported_cat_features = [
            feature_name for feature_name in self.cat_features if feature_name in exported_feature_names
        ]
        exported_num_features = [
            feature_name for feature_name in self.num_features if feature_name in exported_feature_names
        ]

        artifact = {
            "artifact_version": (
                MODEL_ARTIFACT_VERSION_WITH_NOOP_EDITS
                if include_shape_function_edits
                else MODEL_ARTIFACT_VERSION
            ),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "IGANN",
            "model_source": self.model_source,
            "igann_params": igann_params,
            "preprocessing_scope": "whole_dataset_before_split" if self.model_source == "trained" else "unknown",
            "effective_boosting_rounds": int(len(getattr(self.model, "boosting_rates", []) or [])),
            "base_prediction_offset": float(self._infer_base_prediction_offset()),
            "feature_names": exported_feature_names,
            "selected_feature_columns": list(self.selected_feature_columns),
            "target_column": self.target_column,
            "feature_schema": exported_feature_schema,
            "cat_features": exported_cat_features,
            "num_features": exported_num_features,
            "missing_indicator_map": _json_safe(self.missing_indicator_map),
            "numeric_missing_placeholder_values": _json_safe(
                self.numeric_missing_placeholder_values
            ),
            "feature_chart_settings": _json_safe(
                {
                    feature_name: {
                        key: value
                        for key, value in (setting or {}).items()
                        if key not in {"display_title", "x_axis_label"}
                    }
                    for feature_name, setting in self.feature_chart_settings.items()
                }
            ),
            "show_missing_bars": bool(self.show_missing_bars),
            "show_competence_levels": bool(self.show_competence_levels),
            "shape_functions": shape_functions,
            "dataset_id": self.active_dataset_id,
            "dataset_name": self.active_dataset_name
            or (Path(self.active_dataset_path).name if self.active_dataset_path else None),
        }
        held_out_test_data = self._export_held_out_test_data()
        if held_out_test_data is not None:
            artifact["held_out_test_data"] = held_out_test_data
        training_feature_rows = self._export_training_feature_rows()
        if training_feature_rows is not None:
            artifact["training_feature_rows"] = training_feature_rows
            if self.X_train is not None and len(training_feature_rows) == len(self.X_train):
                artifact["training_sample_ids"] = [str(index) for index in self.X_train.index]
        if held_out_test_data is not None and (self.active_dataset_id or self.active_dataset_name):
            artifact["sample_id_namespace"] = str(
                self.active_dataset_id or self.active_dataset_name
            )
        if include_shape_function_edits:
            artifact["shape_function_edits_export"] = {}
        return artifact

    @staticmethod
    def _validate_shape_function_artifact(shape_function: Dict[str, Any]) -> Dict[str, Any]:
        feature_name = str(shape_function.get("feature_name", "")).strip()
        if not feature_name:
            raise ValueError("Imported artifact contains a shape function without feature_name")

        feature_type = str(shape_function.get("feature_type", "")).strip()
        if feature_type not in {"numeric", "categorical"}:
            raise ValueError(f"Invalid feature_type for '{feature_name}'")

        x_values = shape_function.get("x_values")
        y_values = shape_function.get("y_values")
        if not isinstance(x_values, list) or not isinstance(y_values, list):
            raise ValueError(f"Shape function '{feature_name}' must include x_values and y_values arrays")
        if len(x_values) != len(y_values) or not x_values:
            raise ValueError(f"Shape function '{feature_name}' must have equally sized non-empty x/y arrays")

        normalized = {
            "feature_name": feature_name,
            "x_values": [str(x) for x in x_values] if feature_type == "categorical" else [float(x) for x in x_values],
            "y_values": [float(y) for y in y_values],
            "feature_type": feature_type,
            "chart_config": shape_function.get("chart_config") or {},
        }
        distribution = shape_function.get("distribution")
        if distribution is not None:
            if not isinstance(distribution, dict):
                raise ValueError(
                    f"Shape function '{feature_name}' distribution must be an object"
                )
            chart_type = str(distribution.get("chart_type", "")).strip()
            if chart_type not in {"numeric", "categorical"}:
                raise ValueError(
                    f"Shape function '{feature_name}' distribution has an invalid chart_type"
                )
            bins = distribution.get("bins", [])
            counts = distribution.get("counts", [])
            if not isinstance(bins, list) or not isinstance(counts, list):
                raise ValueError(
                    f"Shape function '{feature_name}' distribution bins and counts must be arrays"
                )
            normalized["distribution"] = _json_safe(distribution)
        x_tick_labels = shape_function.get("x_tick_labels")
        if isinstance(x_tick_labels, list):
            normalized["x_tick_labels"] = [str(label) for label in x_tick_labels]
        return normalized

    def _normalize_imported_public_shape_function(
        self,
        shape_function: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_shape = _json_safe(shape_function)

        feature_name = str(normalized_shape.get("feature_name", "")).strip()
        if not feature_name:
            return normalized_shape

        stored_chart_config = _json_safe(self.feature_chart_settings.get(feature_name) or {})
        chart_config = _json_safe(normalized_shape.get("chart_config") or {})
        if not str(chart_config.get("display_title", "") or "").strip():
            chart_config["display_title"] = stored_chart_config.get("display_title", "")
        if not str(chart_config.get("x_axis_label", "") or "").strip():
            chart_config["x_axis_label"] = stored_chart_config.get("x_axis_label", "")
        if not (chart_config.get("categorical_value_labels") or {}):
            chart_config["categorical_value_labels"] = stored_chart_config.get(
                "categorical_value_labels",
                {},
            )
        if not (chart_config.get("numeric_tick_labels") or {}):
            chart_config["numeric_tick_labels"] = stored_chart_config.get(
                "numeric_tick_labels",
                {},
            )
        if "treat_as_categorical" not in chart_config:
            chart_config["treat_as_categorical"] = stored_chart_config.get(
                "treat_as_categorical",
                False,
            )
        if "treat_as_numeric" not in chart_config:
            chart_config["treat_as_numeric"] = stored_chart_config.get(
                "treat_as_numeric",
                False,
            )
        normalized_shape["chart_config"] = chart_config

        if (
            str(normalized_shape.get("feature_type")) != "numeric"
            or str(chart_config.get("chart_feature_type", "numeric")) != "numeric"
        ):
            return normalized_shape

        x_values = normalized_shape.get("x_values") or []
        y_values = normalized_shape.get("y_values") or []
        if (
            not isinstance(x_values, list)
            or not isinstance(y_values, list)
            or len(x_values) != len(y_values)
            or not x_values
        ):
            return normalized_shape

        try:
            x_numeric = np.asarray(x_values, dtype=float)
            y_numeric = np.asarray(y_values, dtype=float)
        except (TypeError, ValueError):
            return normalized_shape

        if x_numeric.size == 0 or y_numeric.size == 0 or x_numeric.size != y_numeric.size:
            return normalized_shape

        order = np.argsort(x_numeric)
        x_numeric = x_numeric[order]
        y_numeric = y_numeric[order]

        schema = self.feature_schema_map.get(feature_name, {}) or {}
        schema_min = schema.get("min_value")
        schema_max = schema.get("max_value")
        try:
            domain_min = float(schema_min)
            domain_max = float(schema_max)
        except (TypeError, ValueError):
            return normalized_shape

        if (
            not np.isfinite(domain_min)
            or not np.isfinite(domain_max)
            or domain_max < domain_min
        ):
            return normalized_shape

        if abs(domain_max - domain_min) < 1e-12:
            target_x = np.asarray([domain_min], dtype=float)
        else:
            target_x = np.linspace(domain_min, domain_max, 30)

        normalized_shape["x_values"] = [float(value) for value in target_x.tolist()]
        normalized_shape["y_values"] = [
            float(value) for value in np.interp(target_x, x_numeric, y_numeric).tolist()
        ]

        numeric_tick_labels = chart_config.get("numeric_tick_labels") or {}
        x_tick_labels = self._build_numeric_chart_tick_labels(
            normalized_shape["x_values"],
            numeric_tick_labels,
        )
        if x_tick_labels is not None:
            normalized_shape["x_tick_labels"] = x_tick_labels
        else:
            normalized_shape.pop("x_tick_labels", None)

        return normalized_shape

    @staticmethod
    def _infer_missing_indicator_map_from_artifact(
        feature_names: List[str],
        selected_feature_columns: List[str],
    ) -> Dict[str, str]:
        selected_feature_set = {
            str(feature_name).strip()
            for feature_name in selected_feature_columns
            if str(feature_name).strip()
        }
        inferred: Dict[str, str] = {}

        for feature_name in feature_names:
            normalized_name = str(feature_name).strip()
            if not normalized_name.endswith(MISSING_INDICATOR_SUFFIX):
                continue
            base_feature_name = normalized_name[: -len(MISSING_INDICATOR_SUFFIX)].strip()
            if base_feature_name and base_feature_name in selected_feature_set:
                inferred[base_feature_name] = normalized_name

        return inferred

    @staticmethod
    def _validate_shape_function_edits_export(
        payload: Dict[str, Any],
        *,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Validate optional exported per-user shape-function edits."""
        included = bool(payload.get("included"))
        users_payload = payload.get("users", [])
        edits_payload = payload.get("edits", [])

        if not isinstance(users_payload, list) or not isinstance(edits_payload, list):
            raise ValueError(
                "Imported artifact shape_function_edits_export must include users and edits arrays"
            )

        normalized_users = []
        declared_user_names = set()
        for user in users_payload:
            if not isinstance(user, dict):
                raise ValueError("Imported artifact shape_function_edits_export users must be objects")
            user_name = str(user.get("name", "")).strip()
            if not user_name:
                raise ValueError("Imported artifact shape_function_edits_export contains a user without a name")
            if user_name in declared_user_names:
                raise ValueError(
                    f"Imported artifact shape_function_edits_export contains duplicate user '{user_name}'"
                )
            declared_user_names.add(user_name)
            normalized_users.append(
                {
                    "name": user_name,
                    "is_superadmin": False,
                }
            )

        normalized_edits = []
        submission_count = 0
        for user_group in edits_payload:
            if not isinstance(user_group, dict):
                raise ValueError(
                    "Imported artifact shape_function_edits_export edits must group entries by user"
                )
            user_name = str(user_group.get("user_name", "")).strip()
            if not user_name:
                raise ValueError(
                    "Imported artifact shape_function_edits_export contains an edit group without user_name"
                )
            if declared_user_names and user_name not in declared_user_names:
                raise ValueError(
                    f"Imported artifact shape_function_edits_export references undeclared user '{user_name}'"
                )

            shape_functions = user_group.get("shape_functions", [])
            if not isinstance(shape_functions, list):
                raise ValueError(
                    f"Imported artifact shape_function_edits_export for '{user_name}' must include shape_functions"
                )

            normalized_shape_functions = []
            for shape_function in shape_functions:
                if not isinstance(shape_function, dict):
                    raise ValueError(
                        "Imported artifact shape_function_edits_export shape function entries must be objects"
                    )
                feature_name = str(shape_function.get("feature_name", "")).strip()
                feature_type = str(shape_function.get("feature_type", "")).strip()
                if feature_name not in feature_names:
                    raise ValueError(
                        f"Imported artifact shape_function_edits_export references unknown feature '{feature_name}'"
                    )
                if feature_type not in {"numeric", "categorical"}:
                    raise ValueError(
                        f"Imported artifact shape_function_edits_export has invalid feature_type for '{feature_name}'"
                    )

                edited_points = shape_function.get("edited_points", [])
                if not isinstance(edited_points, list):
                    raise ValueError(
                        f"Imported artifact shape_function_edits_export for '{feature_name}' must include edited_points"
                    )

                normalized_points = []
                for point in edited_points:
                    if not isinstance(point, dict):
                        raise ValueError(
                            f"Imported artifact shape_function_edits_export points for '{feature_name}' must be objects"
                        )
                    normalized_point = {
                        "x_value": point.get("x_value"),
                        "y_value": float(point.get("y_value")),
                        "weight": float(point.get("weight", 0.5)),
                        "message": str(point.get("message", "") or ""),
                    }
                    if point.get("created_at"):
                        normalized_point["created_at"] = str(point.get("created_at"))
                    if point.get("updated_at"):
                        normalized_point["updated_at"] = str(point.get("updated_at"))
                    normalized_points.append(normalized_point)

                normalized_shape = {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "submission_id": str(shape_function.get("submission_id", "") or ""),
                    "message": str(shape_function.get("message", "") or ""),
                    "is_imputed_noop": bool(shape_function.get("is_imputed_noop", False)),
                    "edited_points": normalized_points,
                }
                if shape_function.get("created_at"):
                    normalized_shape["created_at"] = str(shape_function.get("created_at"))
                if shape_function.get("updated_at"):
                    normalized_shape["updated_at"] = str(shape_function.get("updated_at"))
                if shape_function.get("sureness") is not None:
                    normalized_shape["sureness"] = int(shape_function.get("sureness"))
                normalized_shape_functions.append(normalized_shape)
                submission_count += 1

            normalized_edits.append(
                {
                    "user_name": user_name,
                    "shape_functions": normalized_shape_functions,
                }
            )

        return {
            "included": included,
            "users": normalized_users,
            "edits": normalized_edits,
            "user_count": len(normalized_users),
            "submission_count": submission_count,
        }

    @staticmethod
    def _validate_held_out_test_data(
        raw_test_data: Any,
        *,
        feature_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Validate optional test rows embedded in an exported model artifact."""
        if raw_test_data is None:
            return None
        if not isinstance(raw_test_data, dict):
            raise ValueError("Imported artifact held_out_test_data must be an object")

        feature_rows = raw_test_data.get("feature_rows")
        target_values = raw_test_data.get("target_values")
        if not isinstance(feature_rows, list) or not isinstance(target_values, list):
            raise ValueError(
                "Imported artifact held_out_test_data must include feature_rows and target_values arrays"
            )
        if not feature_rows or len(feature_rows) != len(target_values):
            raise ValueError(
                "Imported artifact held_out_test_data must contain equally sized non-empty rows and targets"
            )

        expected_features = set(feature_names)
        normalized_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(feature_rows):
            if not isinstance(row, dict) or set(row) != expected_features:
                raise ValueError(
                    "Imported artifact held_out_test_data row "
                    f"{index} must contain exactly the model feature_names"
                )
            normalized_rows.append(_json_safe(row))

        normalized_targets: List[float] = []
        for index, value in enumerate(target_values):
            try:
                target = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Imported artifact held_out_test_data target_values must be numeric "
                    f"(invalid value at index {index})"
                ) from exc
            if not np.isfinite(target):
                raise ValueError(
                    "Imported artifact held_out_test_data target_values must be finite "
                    f"(invalid value at index {index})"
                )
            normalized_targets.append(target)

        sample_ids = raw_test_data.get("sample_ids")
        if sample_ids is not None:
            if (
                not isinstance(sample_ids, list)
                or len(sample_ids) != len(feature_rows)
                or any(not str(value).strip() for value in sample_ids)
                or len({str(value) for value in sample_ids}) != len(sample_ids)
            ):
                raise ValueError("Imported artifact held_out_test_data sample_ids must be unique and match the rows")

        result = {
            "feature_rows": normalized_rows,
            "target_values": normalized_targets,
        }
        if sample_ids is not None:
            result["sample_ids"] = [str(value) for value in sample_ids]
        return result

    @staticmethod
    def _validate_training_feature_rows(
        raw_rows: Any,
        *,
        feature_names: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        if raw_rows is None:
            return None
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("Imported artifact training_feature_rows must be a non-empty array")
        expected_features = set(feature_names)
        normalized_rows = []
        for index, row in enumerate(raw_rows):
            if not isinstance(row, dict) or set(row) != expected_features:
                raise ValueError(
                    "Imported artifact training_feature_rows row "
                    f"{index} must contain exactly the model feature_names"
                )
            normalized_rows.append(_json_safe(row))
        return normalized_rows

    def _validate_import_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = [
            "artifact_version",
            "model_type",
            "igann_params",
            "effective_boosting_rounds",
            "base_prediction_offset",
            "feature_names",
            "selected_feature_columns",
            "target_column",
            "feature_schema",
            "cat_features",
            "num_features",
            "feature_chart_settings",
            "shape_functions",
        ]
        missing = [key for key in required_keys if key not in artifact]
        if missing:
            raise ValueError(f"Imported artifact is missing required keys: {missing}")

        artifact_version = str(artifact.get("artifact_version", "")).strip()
        if artifact_version not in SUPPORTED_MODEL_ARTIFACT_VERSIONS:
            raise ValueError(
                "Unsupported artifact version "
                f"'{artifact_version}'. Expected one of {sorted(SUPPORTED_MODEL_ARTIFACT_VERSIONS)}"
            )
        if str(artifact.get("model_type", "")).strip() != "IGANN":
            raise ValueError("Only IGANN model artifacts are supported")

        feature_names = artifact.get("feature_names")
        selected_feature_columns = artifact.get("selected_feature_columns")
        feature_schema = artifact.get("feature_schema")
        cat_features = artifact.get("cat_features")
        num_features = artifact.get("num_features")
        shape_functions = artifact.get("shape_functions")
        missing_indicator_map = artifact.get("missing_indicator_map")
        numeric_missing_placeholder_values = artifact.get(
            "numeric_missing_placeholder_values"
        )

        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("Imported artifact must include non-empty feature_names")
        if not isinstance(selected_feature_columns, list) or not selected_feature_columns:
            raise ValueError("Imported artifact must include non-empty selected_feature_columns")
        if not isinstance(cat_features, list) or not isinstance(num_features, list):
            raise ValueError("Imported artifact must include cat_features and num_features arrays")
        normalized_feature_names = [str(item) for item in feature_names]
        selected_feature_names = [str(item) for item in selected_feature_columns]
        if sorted([str(item) for item in cat_features] + [str(item) for item in num_features]) != sorted(normalized_feature_names):
            raise ValueError("Imported artifact cat_features and num_features must cover all feature_names exactly")
        if not isinstance(shape_functions, list) or len(shape_functions) != len(feature_names):
            raise ValueError("Imported artifact must include one shape function per feature")
        if not set(selected_feature_names).issubset(set(normalized_feature_names)):
            raise ValueError(
                "Imported artifact selected_feature_columns must be covered by feature_names"
            )
        if not isinstance(feature_schema, list) or not feature_schema:
            raise ValueError("Imported artifact must include non-empty feature_schema")

        feature_schema_map = {}
        for item in feature_schema:
            if not isinstance(item, dict):
                raise ValueError("Imported artifact feature_schema entries must be objects")
            feature_name = str(item.get("name", "")).strip()
            feature_type = str(item.get("feature_type", "")).strip()
            if feature_name not in selected_feature_names:
                if feature_name in normalized_feature_names:
                    continue
                raise ValueError(
                    f"Imported artifact schema references unknown feature '{feature_name}'"
                )
            if feature_type not in {"numeric", "categorical"}:
                raise ValueError(f"Imported artifact schema has invalid feature_type for '{feature_name}'")
            feature_schema_map[feature_name] = _json_safe(item)

        if sorted(feature_schema_map.keys()) != sorted(selected_feature_names):
            raise ValueError(
                "Imported artifact feature_schema must cover all selected_feature_columns exactly"
            )

        if missing_indicator_map is None:
            normalized_missing_indicator_map = (
                self._infer_missing_indicator_map_from_artifact(
                    normalized_feature_names,
                    selected_feature_names,
                )
            )
        else:
            if not isinstance(missing_indicator_map, dict):
                raise ValueError(
                    "Imported artifact missing_indicator_map must be an object when present"
                )
            normalized_missing_indicator_map = {}
            for raw_feature_name, raw_indicator_name in missing_indicator_map.items():
                feature_name = str(raw_feature_name).strip()
                indicator_name = str(raw_indicator_name).strip()
                if feature_name not in selected_feature_names:
                    raise ValueError(
                        f"Imported artifact missing_indicator_map references unknown feature '{feature_name}'"
                    )
                if indicator_name not in normalized_feature_names:
                    raise ValueError(
                        f"Imported artifact missing_indicator_map references unknown indicator '{indicator_name}'"
                    )
                normalized_missing_indicator_map[feature_name] = indicator_name

        for feature_name, indicator_name in normalized_missing_indicator_map.items():
            if indicator_name in selected_feature_names:
                raise ValueError(
                    f"Imported artifact missing indicator '{indicator_name}' must not be a public feature"
                )
            if str(feature_schema_map.get(feature_name, {}).get("feature_type")) != "numeric":
                raise ValueError(
                    f"Imported artifact missing indicator '{indicator_name}' must map to a numeric feature"
                )

        allowed_internal_features = set(selected_feature_names) | set(
            normalized_missing_indicator_map.values()
        )
        unexpected_internal_features = sorted(
            feature_name
            for feature_name in normalized_feature_names
            if feature_name not in allowed_internal_features
        )
        if unexpected_internal_features:
            raise ValueError(
                "Imported artifact feature_names contains unsupported internal features: "
                f"{unexpected_internal_features}"
            )

        if not set(normalized_missing_indicator_map.values()).issubset(
            set(str(item) for item in cat_features)
        ):
            raise ValueError(
                "Imported artifact missing indicators must be listed in cat_features"
            )

        normalized_shape_functions = [
            self._validate_shape_function_artifact(item)
            for item in shape_functions
        ]
        if sorted(sf["feature_name"] for sf in normalized_shape_functions) != sorted(normalized_feature_names):
            raise ValueError("Imported artifact shape_functions must cover all feature_names exactly")

        public_shape_functions = [
            shape_function
            for shape_function in normalized_shape_functions
            if shape_function["feature_name"] in selected_feature_names
        ]
        if sorted(sf["feature_name"] for sf in public_shape_functions) != sorted(
            selected_feature_names
        ):
            raise ValueError(
                "Imported artifact shape_functions must cover all selected_feature_columns exactly"
            )

        if numeric_missing_placeholder_values is not None and not isinstance(
            numeric_missing_placeholder_values,
            dict,
        ):
            raise ValueError(
                "Imported artifact numeric_missing_placeholder_values must be an object when present"
            )
        normalized_numeric_missing_placeholder_values: Dict[str, float] = {}
        for feature_name in normalized_missing_indicator_map:
            fallback_value = feature_schema_map.get(feature_name, {}).get(
                "default_value",
                0.0,
            )
            raw_value = (
                numeric_missing_placeholder_values.get(feature_name, fallback_value)
                if isinstance(numeric_missing_placeholder_values, dict)
                else fallback_value
            )
            try:
                normalized_numeric_missing_placeholder_values[feature_name] = float(
                    raw_value
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Imported artifact numeric_missing_placeholder_values must be numeric "
                    f"for '{feature_name}'"
                ) from exc

        shape_function_edits_export = artifact.get("shape_function_edits_export")
        if shape_function_edits_export is not None and not isinstance(
            shape_function_edits_export,
            dict,
        ):
            raise ValueError(
                "Imported artifact shape_function_edits_export must be an object when present"
            )

        normalized_shape_function_edits_export = self._validate_shape_function_edits_export(
            _json_safe(shape_function_edits_export or {}),
            feature_names=normalized_feature_names,
        ) if shape_function_edits_export is not None else {}
        held_out_test_data = self._validate_held_out_test_data(
            artifact.get("held_out_test_data"),
            feature_names=normalized_feature_names,
        )
        training_feature_rows = self._validate_training_feature_rows(
            artifact.get("training_feature_rows"),
            feature_names=normalized_feature_names,
        )
        # Older exports round the schema default (and sometimes the explicit
        # placeholder metadata) to two decimals. The exported, preprocessed
        # training rows retain the exact fitted fill used by the model.
        for feature_name, indicator_name in normalized_missing_indicator_map.items():
            missing_rows = [
                row for row in (training_feature_rows or [])
                if row.get(indicator_name) == MISSING_CATEGORY_VALUE
            ]
            if not missing_rows:
                continue
            try:
                fills = [float(row[feature_name]) for row in missing_rows]
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(value) and value == fills[0] for value in fills):
                continue
            declared = normalized_numeric_missing_placeholder_values[feature_name]
            if (
                np.isclose(declared, fills[0], rtol=0, atol=1e-8)
                or np.isclose(declared, round(fills[0], 2), rtol=0, atol=1e-8)
            ):
                normalized_numeric_missing_placeholder_values[feature_name] = fills[0]
        training_sample_ids = artifact.get("training_sample_ids")
        if training_sample_ids is not None:
            if (
                not isinstance(training_sample_ids, list)
                or any(not str(value).strip() for value in training_sample_ids)
                or len({str(value) for value in training_sample_ids}) != len(training_sample_ids)
                or (training_feature_rows is not None and len(training_sample_ids) != len(training_feature_rows))
            ):
                raise ValueError("Imported artifact training_sample_ids must be unique and match training rows")
            training_sample_ids = [str(value) for value in training_sample_ids]
        sample_id_namespace = artifact.get("sample_id_namespace")
        if sample_id_namespace is not None:
            sample_id_namespace = str(sample_id_namespace).strip()
            if not sample_id_namespace:
                raise ValueError("Imported artifact sample_id_namespace must be non-empty")
        preprocessing_scope = artifact.get("preprocessing_scope")
        if preprocessing_scope is not None and preprocessing_scope not in {
            "whole_dataset_before_split", "training_partition_only", "unknown"
        }:
            raise ValueError("Imported artifact preprocessing_scope is invalid")

        return {
            "artifact_version": artifact_version,
            "igann_params": dict(artifact.get("igann_params") or {}),
            "effective_boosting_rounds": int(artifact.get("effective_boosting_rounds") or 0),
            "base_prediction_offset": float(artifact.get("base_prediction_offset")),
            "feature_names": normalized_feature_names,
            "selected_feature_columns": selected_feature_names,
            "target_column": str(artifact.get("target_column")),
            "feature_schema": list(feature_schema_map.values()),
            "feature_schema_map": feature_schema_map,
            "cat_features": [str(item) for item in cat_features],
            "num_features": [str(item) for item in num_features],
            "missing_indicator_map": normalized_missing_indicator_map,
            "numeric_missing_placeholder_values": (
                normalized_numeric_missing_placeholder_values
            ),
            "feature_chart_settings": {
                str(feature_name): normalized
                for feature_name, raw_setting in (
                    _json_safe(artifact.get("feature_chart_settings") or {})
                ).items()
                for normalized in [self._normalize_stored_feature_chart_setting(raw_setting)]
                if normalized
            },
            "show_missing_bars": bool(artifact.get("show_missing_bars", False)),
            "show_competence_levels": bool(
                artifact.get("show_competence_levels", False)
            ),
            "shape_functions": normalized_shape_functions,
            "public_shape_functions": public_shape_functions,
            "dataset_id": artifact.get("dataset_id"),
            "dataset_name": artifact.get("dataset_name"),
            "shape_function_edits_export": normalized_shape_function_edits_export,
            "held_out_test_data": held_out_test_data,
            "training_feature_rows": training_feature_rows,
            "training_sample_ids": training_sample_ids,
            "sample_id_namespace": sample_id_namespace,
            "preprocessing_scope": preprocessing_scope,
        }

    def import_model_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        validated = self._validate_import_artifact(artifact)
        imported_model = ImportedIGANNRuntime(
            feature_names=validated["feature_names"],
            shape_functions=validated["shape_functions"],
            base_prediction_offset=validated["base_prediction_offset"],
            igann_params=validated["igann_params"],
            effective_boosting_rounds=validated["effective_boosting_rounds"],
        )

        self.model = imported_model
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.imported_training_feature_rows = validated["training_feature_rows"]
        self.df = None
        self.is_trained = True
        self.feature_names = list(validated["feature_names"])
        self.selected_feature_columns = list(validated["selected_feature_columns"])
        self.cat_features = list(validated["cat_features"])
        self.num_features = list(validated["num_features"])
        self.target_column = validated["target_column"]
        self.feature_schema = list(validated["feature_schema"])
        self.feature_schema_map = dict(validated["feature_schema_map"])
        self.missing_indicator_map = dict(validated["missing_indicator_map"])
        self.numeric_missing_placeholder_values = dict(
            validated["numeric_missing_placeholder_values"]
        )
        self.feature_chart_settings = dict(validated["feature_chart_settings"])
        self.show_missing_bars = bool(validated.get("show_missing_bars", False))
        self.show_competence_levels = bool(
            validated.get("show_competence_levels", False)
        )
        self.original_shape_functions = {
            shape_function["feature_name"]: self._normalize_imported_public_shape_function(
                shape_function
            )
            for shape_function in validated["public_shape_functions"]
        }
        self.shape_function_offsets = {}
        self.model_source = "imported"
        self.imported_artifact_version = validated["artifact_version"]
        self.analytics_available = False
        self.primary_n_estimators = None
        self.active_dataset_id = None
        self.active_dataset_path = None
        self.active_dataset_name = None
        self._clear_comparison_state()
        self.active_dataset_file.unlink(missing_ok=True)

        return {
            "success": True,
            "message": "Model imported successfully",
            "model_source": self.model_source,
            "imported_artifact_version": self.imported_artifact_version,
            "imported_shape_function_edits": bool(
                validated.get("shape_function_edits_export", {}).get("included")
            ),
            "imported_edit_user_count": 0,
            "imported_edit_submission_count": 0,
            "_shape_function_edits_export": validated.get(
                "shape_function_edits_export", {}
            ),
        }
