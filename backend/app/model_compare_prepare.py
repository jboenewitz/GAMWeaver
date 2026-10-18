"""Prepare imported artifacts and held-out data for isolated comparison."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
import numpy as np

from .data_processing import MISSING_CATEGORY_VALUE
from .ml_shared import ImportedIGANNRuntime, _json_safe


class ModelComparePrepareMixin:
    def _normalize_imported_public_shape_function_for_context(
        self,
        shape_function: Dict[str, Any],
        *,
        feature_chart_settings: Dict[str, Dict[str, Any]],
        feature_schema_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized_shape = _json_safe(shape_function)

        feature_name = str(normalized_shape.get("feature_name", "")).strip()
        if not feature_name:
            return normalized_shape

        stored_chart_config = _json_safe(
            (feature_chart_settings or {}).get(feature_name) or {}
        )
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

        schema = (feature_schema_map or {}).get(feature_name, {}) or {}
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
    def _format_compare_numeric_display(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _decode_compare_point_display_value(
        cls,
        feature_type: str,
        raw_value: Any,
        original_x: List[Any],
    ) -> Any:
        if feature_type == "categorical":
            return str(raw_value)
        decoded = cls._decode_numeric_storage_key(raw_value, original_x)
        if decoded is not None:
            return float(decoded)
        return raw_value

    @classmethod
    def _compare_point_sort_key(
        cls,
        feature_type: str,
        raw_value: Any,
        original_x: List[Any],
    ) -> Any:
        if feature_type == "categorical":
            return (0, str(raw_value))
        decoded = cls._decode_numeric_storage_key(raw_value, original_x)
        if decoded is not None:
            return (0, float(decoded))
        try:
            return (0, float(raw_value))
        except (TypeError, ValueError):
            return (1, str(raw_value))

    @classmethod
    def _build_compare_x_summary(
        cls,
        feature_type: str,
        edited_points: List[Dict[str, Any]],
        original_x: List[Any],
    ) -> str:
        if not edited_points:
            return ""

        if feature_type == "categorical":
            labels = [str(point.get("display_x_value", point.get("x_value"))) for point in edited_points]
            unique_labels = list(dict.fromkeys(labels))
            if len(unique_labels) <= 3:
                return ", ".join(unique_labels)
            preview = ", ".join(unique_labels[:3])
            return f"{preview}, +{len(unique_labels) - 3} more"

        numeric_values: List[float] = []
        for point in edited_points:
            decoded = cls._decode_numeric_storage_key(point.get("x_value"), original_x)
            if decoded is None:
                continue
            numeric_values.append(float(decoded))

        numeric_values = sorted(numeric_values)
        if not numeric_values:
            return str(edited_points[0].get("display_x_value", edited_points[0].get("x_value", "")))
        if len(numeric_values) == 1:
            return cls._format_compare_numeric_display(numeric_values[0])
        if len(numeric_values) == 2:
            return ", ".join(
                cls._format_compare_numeric_display(value) for value in numeric_values
            )
        return (
            f"{cls._format_compare_numeric_display(numeric_values[0])}"
            f" to {cls._format_compare_numeric_display(numeric_values[-1])}"
        )

    @classmethod
    def _decode_numeric_offsets_for_shape_function(
        cls,
        shape_function: Dict[str, Any],
        offsets: Dict[Any, float],
    ) -> List[Tuple[float, float]]:
        original_x = shape_function.get("x_values", [])
        point_map: Dict[float, float] = {}
        for raw_key, raw_offset in (offsets or {}).items():
            x_val = cls._decode_numeric_storage_key(raw_key, original_x)
            if x_val is None:
                continue
            try:
                point_map[float(x_val)] = float(raw_offset)
            except (TypeError, ValueError):
                continue
        return sorted(point_map.items(), key=lambda item: item[0])

    @classmethod
    def _get_compare_offset_for_feature_value(
        cls,
        *,
        shape_function: Dict[str, Any],
        original_is_categorical: bool,
        value: Any,
        offsets: Optional[Dict[Any, float]] = None,
    ) -> float:
        offsets = offsets or {}
        if not offsets:
            return 0.0

        is_categorical_chart = str(shape_function.get("feature_type")) == "categorical"
        if is_categorical_chart:
            candidate_keys = [
                cls._stringify_chart_value(value),
                str(value),
            ]
            for raw_key in candidate_keys:
                if raw_key in offsets:
                    try:
                        return float(offsets.get(raw_key, 0.0))
                    except (TypeError, ValueError):
                        return 0.0

            points = cls._decode_numeric_offsets_for_shape_function(shape_function, offsets)
            if points:
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    return 0.0
                for x_val, x_offset in points:
                    if abs(x_val - value_float) < 1e-9:
                        return float(x_offset)
            return 0.0

        if original_is_categorical:
            candidate_keys = [
                cls._stringify_chart_value(value),
                str(value),
            ]
            for raw_key in candidate_keys:
                if raw_key in offsets:
                    try:
                        return float(offsets.get(raw_key, 0.0))
                    except (TypeError, ValueError):
                        return 0.0

        points = cls._decode_numeric_offsets_for_shape_function(shape_function, offsets)
        if not points:
            return 0.0

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return 0.0

        if len(points) == 1:
            return float(points[0][1])

        x_positions = [point[0] for point in points]
        offset_values = [point[1] for point in points]

        if value_float <= x_positions[0]:
            return float(offset_values[0])
        if value_float >= x_positions[-1]:
            return float(offset_values[-1])

        for idx in range(len(points) - 1):
            left_x, left_offset = points[idx]
            right_x, right_offset = points[idx + 1]
            if left_x <= value_float <= right_x:
                span = right_x - left_x
                if span == 0:
                    return float(left_offset)
                ratio = (value_float - left_x) / span
                return float(left_offset + ratio * (right_offset - left_offset))

        return float(offset_values[-1])

    def prepare_model_compare_artifact(
        self,
        artifact: Dict[str, Any],
        *,
        filename: str,
    ) -> Dict[str, Any]:
        validated = self._validate_import_artifact(artifact)
        feature_chart_settings = dict(validated["feature_chart_settings"])
        feature_schema_map = dict(validated["feature_schema_map"])

        raw_public_shape_map = {
            shape_function["feature_name"]: shape_function
            for shape_function in validated["public_shape_functions"]
        }
        prepared_shape_functions = []
        prepared_shape_map: Dict[str, Dict[str, Any]] = {}
        for feature_name in validated["selected_feature_columns"]:
            shape_function = raw_public_shape_map.get(feature_name)
            if not shape_function:
                continue
            normalized_shape = self._normalize_imported_public_shape_function_for_context(
                shape_function,
                feature_chart_settings=feature_chart_settings,
                feature_schema_map=feature_schema_map,
            )
            prepared_shape_functions.append(normalized_shape)
            prepared_shape_map[feature_name] = normalized_shape

        # Keep internal missing-indicator shape functions for held-out prediction
        # reconstruction while exposing only public shape functions in the UI.
        prepared_feature_names = {
            str(shape_function.get("feature_name"))
            for shape_function in prepared_shape_functions
        }
        prepared_shape_functions.extend(
            _json_safe(shape_function)
            for shape_function in validated["shape_functions"]
            if str(shape_function.get("feature_name")) not in prepared_feature_names
        )

        submissions_by_feature: Dict[str, Dict[str, Any]] = {
            feature_name: {
                "feature_name": feature_name,
                "feature_type": prepared_shape_map.get(feature_name, {}).get(
                    "feature_type",
                    "numeric",
                ),
                "submissions": [],
            }
            for feature_name in validated["selected_feature_columns"]
        }

        edits_export = validated.get("shape_function_edits_export", {}) or {}
        declared_professions = {
            str(user.get("name", "")).strip(): (
                str(user.get("profession", "")).strip() or None
            )
            for user in (edits_export.get("users") or [])
            if isinstance(user, dict) and str(user.get("name", "")).strip()
        }

        for user_index, user_group in enumerate(edits_export.get("edits") or []):
            user_name = str(user_group.get("user_name", "")).strip()
            if not user_name:
                continue

            for shape_index, shape_function in enumerate(
                user_group.get("shape_functions") or []
            ):
                feature_name = str(shape_function.get("feature_name", "")).strip()
                base_shape = prepared_shape_map.get(feature_name)
                if not feature_name or base_shape is None:
                    continue

                effective_feature_type = str(base_shape.get("feature_type", "numeric"))
                original_x = base_shape.get("x_values", [])
                edited_points = []
                for point in shape_function.get("edited_points") or []:
                    normalized_point = {
                        "x_value": point.get("x_value"),
                        "y_value": float(point.get("y_value", 0.0)),
                        "weight": float(point.get("weight", 0.5)),
                        "message": str(point.get("message", "") or ""),
                    }
                    normalized_point["display_x_value"] = self._decode_compare_point_display_value(
                        effective_feature_type,
                        normalized_point["x_value"],
                        original_x,
                    )
                    edited_points.append(normalized_point)

                if not edited_points:
                    continue

                edited_points.sort(
                    key=lambda point: self._compare_point_sort_key(
                        effective_feature_type,
                        point.get("x_value"),
                        original_x,
                    )
                )

                raw_submission_id = str(shape_function.get("submission_id", "") or "").strip()
                submission_id = (
                    raw_submission_id
                    or f"artifact-{user_index}-{shape_index}-{feature_name}"
                )
                avg_weight = sum(
                    float(point.get("weight", 0.5)) for point in edited_points
                ) / max(len(edited_points), 1)
                sureness = shape_function.get("sureness")
                try:
                    sureness_value = int(sureness)
                except (TypeError, ValueError):
                    sureness_value = round(avg_weight * 10)

                submissions_by_feature[feature_name]["submissions"].append(
                    {
                        "submission_id": submission_id,
                        "feature_name": feature_name,
                        "feature_type": effective_feature_type,
                        "user_name": user_name,
                        "profession": declared_professions.get(user_name),
                        "message": str(shape_function.get("message", "") or ""),
                        "is_imputed_noop": bool(shape_function.get("is_imputed_noop", False)),
                        "sureness": sureness_value,
                        "point_count": len(edited_points),
                        "x_summary": self._build_compare_x_summary(
                            effective_feature_type,
                            edited_points,
                            original_x,
                        ),
                        "created_at": shape_function.get("created_at"),
                        "updated_at": shape_function.get("updated_at"),
                        "edited_points": edited_points,
                    }
                )

        ordered_submissions_by_feature = []
        for feature_name in validated["selected_feature_columns"]:
            feature_group = submissions_by_feature.get(feature_name)
            if feature_group is None:
                continue
            feature_group["submissions"].sort(
                key=lambda submission: (
                    str(submission.get("created_at") or ""),
                    submission.get("submission_id", ""),
                ),
                reverse=True,
            )
            ordered_submissions_by_feature.append(feature_group)

        has_edit_export = bool(edits_export.get("included"))
        metadata = {
            "filename": filename,
            "exported_at": artifact.get("exported_at"),
            "dataset_name": validated.get("dataset_name"),
            "dataset_id": validated.get("dataset_id"),
            "target_column": validated["target_column"],
            "selected_feature_count": len(validated["selected_feature_columns"]),
            "has_edit_export": has_edit_export,
            "edit_user_count": int(edits_export.get("user_count", 0) or 0),
            "edit_submission_count": int(edits_export.get("submission_count", 0) or 0),
            "model_source": str(artifact.get("model_source", "") or ""),
            "artifact_version": validated["artifact_version"],
            "has_held_out_test_data": bool(validated.get("held_out_test_data")),
        }

        return {
            "metadata": metadata,
            "base_prediction_offset": validated["base_prediction_offset"],
            "feature_names": list(validated["feature_names"]),
            "selected_feature_columns": list(validated["selected_feature_columns"]),
            "cat_features": list(validated["cat_features"]),
            "num_features": list(validated["num_features"]),
            "missing_indicator_map": dict(validated["missing_indicator_map"]),
            "numeric_missing_placeholder_values": dict(
                validated["numeric_missing_placeholder_values"]
            ),
            "shape_functions": prepared_shape_functions,
            "submissions_by_feature": ordered_submissions_by_feature,
            "held_out_test_data": validated.get("held_out_test_data"),
            "training_feature_rows": validated.get("training_feature_rows"),
            "training_sample_ids": validated.get("training_sample_ids"),
            "sample_id_namespace": validated.get("sample_id_namespace"),
            "preprocessing_scope": validated.get("preprocessing_scope"),
        }

    @staticmethod
    def _validate_model_compare_pair(
        left_artifact: Dict[str, Any],
        right_artifact: Dict[str, Any],
    ) -> List[str]:
        left_target = str(
            ((left_artifact.get("metadata") or {}).get("target_column")) or ""
        )
        right_target = str(
            ((right_artifact.get("metadata") or {}).get("target_column")) or ""
        )
        if left_target != right_target:
            raise ValueError("Compared artifacts must have the same target_column")

        left_features = [
            str(feature_name)
            for feature_name in (left_artifact.get("selected_feature_columns") or [])
        ]
        right_features = [
            str(feature_name)
            for feature_name in (right_artifact.get("selected_feature_columns") or [])
        ]
        if (
            len(left_features) != len(right_features)
            or set(left_features) != set(right_features)
        ):
            raise ValueError(
                "Compared artifacts must have identical selected_feature_columns"
            )

        left_shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (left_artifact.get("shape_functions") or [])
        }
        right_shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (right_artifact.get("shape_functions") or [])
        }
        for feature_name in left_features:
            left_shape = left_shape_map.get(feature_name) or {}
            right_shape = right_shape_map.get(feature_name) or {}
            left_feature_type = str(
                left_shape.get("feature_type", "")
            )
            right_feature_type = str(
                right_shape.get("feature_type", "")
            )
            if left_feature_type != right_feature_type:
                raise ValueError(
                    f"Compared artifacts must render feature '{feature_name}' with the same chart type"
                )
            left_x_values = list(left_shape.get("x_values") or [])
            right_x_values = list(right_shape.get("x_values") or [])
            if left_feature_type == "categorical":
                if len(left_x_values) != len(right_x_values):
                    raise ValueError(
                        f"Compared artifacts must use aligned x_values for categorical feature '{feature_name}'"
                    )
                if left_x_values != right_x_values:
                    raise ValueError(
                        f"Compared artifacts must use the same x_values order for categorical feature '{feature_name}'"
                    )

        return left_features

    def _predict_model_compare_artifact_rows(
        self,
        artifact: Dict[str, Any],
        feature_rows: List[Dict[str, Any]],
        *,
        offsets: Optional[Dict[str, Dict[Any, float]]] = None,
        feature_shifts: Optional[Dict[str, float]] = None,
        intercept_adjustment: float = 0.0,
        exclude_missing_indicators: bool = False,
    ) -> np.ndarray:
        """Reconstruct predictions from one prepared artifact's shape functions."""
        missing_indicator_features = (
            self._compare_missing_indicator_features(artifact)
            if exclude_missing_indicators
            else set()
        )
        shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (artifact.get("shape_functions") or [])
        }
        categorical_features = {
            str(feature_name) for feature_name in (artifact.get("cat_features") or [])
        }
        predictions: List[float] = []
        for row in feature_rows:
            prediction = (
                float(artifact.get("base_prediction_offset", 0.0) or 0.0)
                + intercept_adjustment
            )
            for feature_name in artifact.get("feature_names") or []:
                if feature_name in missing_indicator_features:
                    continue
                shape_function = shape_map.get(str(feature_name))
                if shape_function is None:
                    continue
                value = row.get(feature_name)
                if str(shape_function.get("feature_type")) == "categorical":
                    prediction += ImportedIGANNRuntime._predict_categorical(
                        shape_function,
                        value,
                    )
                else:
                    prediction += ImportedIGANNRuntime._predict_numeric(
                        shape_function,
                        value,
                    )
                prediction += self._get_compare_offset_for_feature_value(
                    shape_function=shape_function,
                    original_is_categorical=feature_name in categorical_features,
                    value=value,
                    offsets=(offsets or {}).get(feature_name, {}),
                )
                prediction += float((feature_shifts or {}).get(feature_name, 0.0))
            predictions.append(float(prediction))
        return np.asarray(predictions, dtype=float)

    @staticmethod
    def _project_compare_rows(
        feature_rows: List[Dict[str, Any]],
        source_artifact: Dict[str, Any],
        target_artifact: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Apply the target export's fitted fills to shared preprocessed rows."""
        source_indicators = source_artifact.get("missing_indicator_map") or {}
        target_indicators = target_artifact.get("missing_indicator_map") or {}
        target_fills = target_artifact.get("numeric_missing_placeholder_values") or {}
        result = []
        for source_row in feature_rows:
            row = dict(source_row)
            for feature_name, target_indicator in target_indicators.items():
                source_indicator = source_indicators.get(feature_name)
                row[target_indicator] = (
                    MISSING_CATEGORY_VALUE
                    if source_indicator is not None
                    and source_row.get(source_indicator) == MISSING_CATEGORY_VALUE
                    else "Observed"
                )
            for feature_name, source_indicator in source_indicators.items():
                if (
                    source_row.get(source_indicator) == MISSING_CATEGORY_VALUE
                    and feature_name in target_fills
                ):
                    row[feature_name] = target_fills[feature_name]
            result.append({
                name: row.get(name)
                for name in target_artifact.get("feature_names") or []
            })
        return result

    def _build_held_out_test_performance(
        self,
        *,
        original_artifact: Dict[str, Any],
        edited_artifact: Dict[str, Any],
        test_data_artifact: Dict[str, Any],
        test_data_source: str,
        edited_offsets: Dict[str, Dict[Any, float]],
        edited_feature_shifts: Dict[str, float],
        intercept_adjustment: float = 0.0,
        intercept_correction_source: str = "manual",
    ) -> Optional[Dict[str, Any]]:
        """Evaluate full, reduced, and edited models on one shared row set."""
        held_out_test_data = test_data_artifact.get("held_out_test_data") or {}
        feature_rows = held_out_test_data.get("feature_rows") or []
        target_values = held_out_test_data.get("target_values") or []
        if not feature_rows or len(feature_rows) != len(target_values):
            return None

        actual_values = np.asarray(target_values, dtype=float)
        test_sample_ids = held_out_test_data.get("sample_ids") or []
        test_namespace = test_data_artifact.get("sample_id_namespace")
        def verify_test_exclusion(artifact: Dict[str, Any], label: str) -> bool:
            train_ids = artifact.get("training_sample_ids") or []
            if not (test_sample_ids and test_namespace and train_ids
                    and artifact.get("sample_id_namespace") == test_namespace):
                return False
            overlap = set(test_sample_ids).intersection(train_ids)
            if overlap:
                raise ValueError(
                    f"{label} training data contains {len(overlap)} selected test sample IDs"
                )
            return True
        full_exclusion_verified = verify_test_exclusion(original_artifact, "Full-data")
        reduced_exclusion_verified = verify_test_exclusion(edited_artifact, "Reduced-data")
        def rows_for(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Recreate model-specific missing placeholders from shared row semantics."""
            if artifact is test_data_artifact or (
                test_data_source == "main_training" and artifact is original_artifact
            ):
                return [
                    {name: source_row.get(name) for name in artifact.get("feature_names") or []}
                    for source_row in feature_rows
                ]
            return self._project_compare_rows(
                feature_rows, test_data_artifact, artifact,
            )
        full_predictions = self._predict_model_compare_artifact_rows(
            original_artifact,
            rows_for(original_artifact),
        )
        reduced_predictions = self._predict_model_compare_artifact_rows(
            edited_artifact,
            rows_for(edited_artifact),
        )
        edited_predictions = self._predict_model_compare_artifact_rows(
            edited_artifact,
            rows_for(edited_artifact),
            offsets=edited_offsets,
            feature_shifts=edited_feature_shifts,
            intercept_adjustment=intercept_adjustment,
        )
        def evaluate(predictions: np.ndarray) -> Dict[str, Optional[float]]:
            # Positive bias means overprediction; negative bias means underprediction.
            residuals = predictions - actual_values
            return {
                "mae": float(mean_absolute_error(actual_values, predictions)),
                "rmse": float(root_mean_squared_error(actual_values, predictions)),
                # Population SD of predicted values (ddof=0).
                "std_dev": float(np.std(predictions)),
                "mean_error": float(np.mean(residuals)),
                "r2": (
                    float(r2_score(actual_values, predictions))
                    if len(actual_values) >= 2 and np.ptp(actual_values) > 0
                    else None
                ),
            }

        error_change = np.abs(edited_predictions - actual_values) - np.abs(
            reduced_predictions - actual_values
        )
        rng = np.random.default_rng(42)
        bootstrap_differences = {key: [] for key in ("mae", "rmse", "r2")}
        for _ in range(1000):
            sample = rng.integers(0, len(actual_values), len(actual_values))
            sampled_actual = actual_values[sample]
            sampled_reduced = reduced_predictions[sample]
            sampled_edited = edited_predictions[sample]
            bootstrap_differences["mae"].append(
                float(np.mean(np.abs(sampled_edited - sampled_actual))
                      - np.mean(np.abs(sampled_reduced - sampled_actual)))
            )
            bootstrap_differences["rmse"].append(
                float(np.sqrt(np.mean((sampled_edited - sampled_actual) ** 2))
                      - np.sqrt(np.mean((sampled_reduced - sampled_actual) ** 2)))
            )
            if len(sampled_actual) >= 2 and np.ptp(sampled_actual) > 0:
                bootstrap_differences["r2"].append(
                    float(r2_score(sampled_actual, sampled_edited)
                          - r2_score(sampled_actual, sampled_reduced))
                )
        bootstrap_ci = {
            key: (
                [float(value) for value in np.percentile(samples, [2.5, 97.5])]
                if samples else None
            )
            for key, samples in bootstrap_differences.items()
        }

        def matching_training_rows(artifact: Dict[str, Any]) -> int:
            # Feature equality is only a warning signal; distinct people can share values.
            training = artifact.get("training_feature_rows") or []
            signatures = {json.dumps(row, sort_keys=True, default=str) for row in training}
            return sum(
                json.dumps(row, sort_keys=True, default=str) in signatures
                for row in rows_for(artifact)
            )

        reduced_test_data = edited_artifact.get("held_out_test_data") or {}
        same_exported_test_rows = (
            reduced_test_data.get("feature_rows") == feature_rows
            and reduced_test_data.get("target_values") == target_values
        ) if reduced_test_data.get("feature_rows") else None

        return {
            "test_size": len(feature_rows),
            "test_data_source": test_data_source,
            "baseline": evaluate(full_predictions),
            "reduced": evaluate(reduced_predictions),
            "edited": evaluate(edited_predictions),
            "edited_before_intercept_correction": (
                evaluate(edited_predictions - intercept_adjustment)
                if intercept_adjustment else None
            ),
            "actual_values": [float(value) for value in actual_values.tolist()],
            "baseline_predictions": [
                float(value) for value in full_predictions.tolist()
            ],
            "reduced_predictions": [float(value) for value in reduced_predictions.tolist()],
            "edited_predictions": [float(value) for value in edited_predictions.tolist()],
            "edited_before_intercept_predictions": (
                [float(value) for value in (edited_predictions - intercept_adjustment).tolist()]
                if intercept_adjustment else None
            ),
            "edit_effect": {
                "improved_percent": float(100 * np.mean(error_change < 0)),
                "worsened_percent": float(100 * np.mean(error_change > 0)),
                "unchanged_percent": float(100 * np.mean(error_change == 0)),
                "median_absolute_error_change": float(np.median(error_change)),
                "absolute_error_changes": [float(value) for value in error_change.tolist()],
            },
            "paired_bootstrap": {"resamples": 1000, "seed": 42, "ci_95": bootstrap_ci},
            "validation": {
                "same_exported_test_rows": same_exported_test_rows,
                "full_training_feature_matches": matching_training_rows(original_artifact),
                "reduced_training_feature_matches": matching_training_rows(edited_artifact),
                "full_training_rows_available": bool(original_artifact.get("training_feature_rows")),
                "reduced_training_rows_available": bool(edited_artifact.get("training_feature_rows")),
                "sample_ids_available": bool(test_sample_ids),
                "full_test_exclusion_verified": full_exclusion_verified,
                "reduced_test_exclusion_verified": reduced_exclusion_verified,
                "common_holdout_verified": (
                    full_exclusion_verified and reduced_exclusion_verified
                ),
                "preprocessing_scope": {
                    "full": original_artifact.get("preprocessing_scope") or "unknown",
                    "reduced": edited_artifact.get("preprocessing_scope") or "unknown",
                },
            },
            "intercept_correction": {
                "enabled": intercept_correction_source == "reduced training rows" or bool(intercept_adjustment),
                "shift": float(intercept_adjustment),
                "calculated_on": intercept_correction_source,
            },
        }
