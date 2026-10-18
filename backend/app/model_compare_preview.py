"""Build the exported model comparison preview without changing active state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .ml_shared import MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS, _format_compare_chart_trace_user_name, _json_safe


class ModelComparePreviewMixin:
    def build_model_compare_preview(
        self,
        *,
        left_artifact: Dict[str, Any],
        right_artifact: Dict[str, Any],
        left_selected_submission_ids: Optional[List[str]],
        right_selected_submission_ids: Optional[List[str]],
        use_confidence: bool = True,
        limit_edited_mean_deviation: bool = False,
        feature_names: Optional[List[str]] = None,
        overall_mae_feature_names: Optional[List[str]] = None,
        mean_deviation_feature_names: Optional[List[str]] = None,
        mean_deviation_scaled_feature_names: Optional[List[str]] = None,
        manual_feature_shifts: Optional[Dict[str, float]] = None,
        held_out_test_source: str = "right",
        intercept_adjustment: float = 0.0,
        apply_training_intercept_correction: bool = False,
        feature_metrics_on_original_test_set: bool = False,
    ) -> Dict[str, Any]:
        """Compare two prepared exports without mutating the active model.

        The preview combines selected edits, confidence weighting, optional
        mean-deviation limits, feature curves, and held-out performance.
        """
        # Validate the requested comparison and collect selected edit offsets.
        intercept_adjustment = float(intercept_adjustment)
        if not np.isfinite(intercept_adjustment):
            raise ValueError("intercept_adjustment must be finite")
        original_edited_base = float(right_artifact.get("base_prediction_offset", 0.0) or 0.0)
        adjusted_edited_base = original_edited_base + intercept_adjustment
        if not np.isfinite(adjusted_edited_base):
            raise ValueError("adjusted edited base prediction offset must be finite")
        held_out_test_source = str(held_out_test_source or "right").strip().lower()
        if held_out_test_source not in {"left", "right", "main_training"}:
            raise ValueError(
                "held_out_test_source must be 'left', 'right', or 'main_training'"
            )
        original_test_rows = (
            (left_artifact.get("held_out_test_data") or {}).get("feature_rows") or []
        )
        if feature_metrics_on_original_test_set and not original_test_rows:
            raise ValueError(
                "Original artifact has no held-out feature rows for per-feature metrics"
            )
        shared_features = self._validate_model_compare_pair(
            left_artifact,
            right_artifact,
        )
        if feature_names:
            requested = []
            shared_feature_set = set(shared_features)
            for feature_name in feature_names:
                normalized_name = str(feature_name).strip()
                if normalized_name and normalized_name in shared_feature_set:
                    requested.append(normalized_name)
            active_features = requested or shared_features
        else:
            active_features = shared_features
        overall_mae_feature_set = (
            {str(feature_name).strip() for feature_name in overall_mae_feature_names}
            if overall_mae_feature_names is not None
            else set(active_features)
        )
        mean_deviation_feature_set = (
            {
                str(feature_name).strip()
                for feature_name in mean_deviation_feature_names
                if str(feature_name).strip() in active_features
            }
            if mean_deviation_feature_names is not None
            else set(shared_features)
        )
        explicitly_scaled_mean_deviation_feature_set = (
            {
                str(feature_name).strip()
                for feature_name in mean_deviation_scaled_feature_names
                if str(feature_name).strip() in mean_deviation_feature_set
            }
            if mean_deviation_scaled_feature_names is not None
            else set()
        )
        mean_deviation_scaled_feature_set = (
            explicitly_scaled_mean_deviation_feature_set
            or mean_deviation_feature_set
        )
        should_limit_edited_mean_deviation = bool(
            limit_edited_mean_deviation
            or explicitly_scaled_mean_deviation_feature_set
        )
        right_model_feature_names = set(right_artifact.get("feature_names") or [])
        right_manual_feature_shifts = {
            str(feature_name).strip(): float(shift)
            for feature_name, shift in (manual_feature_shifts or {}).items()
            if str(feature_name).strip() in right_model_feature_names
            and np.isfinite(float(shift))
            and abs(float(shift)) > 1e-12
        }

        left_shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (left_artifact.get("shape_functions") or [])
        }
        right_shape_map = {
            str(shape_function.get("feature_name")): shape_function
            for shape_function in (right_artifact.get("shape_functions") or [])
        }
        left_offsets = self._collect_compare_offsets(
            left_artifact,
            left_selected_submission_ids,
            use_confidence=use_confidence,
        )
        right_offsets = self._collect_compare_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=use_confidence,
        )
        alternate_confidence_offsets = self._collect_compare_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=not use_confidence,
        )
        left_submission_offsets = self._collect_compare_submission_offsets(
            left_artifact,
            left_selected_submission_ids,
            use_confidence=use_confidence,
        )
        right_submission_offsets = self._collect_compare_submission_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=use_confidence,
        )
        right_alternate_submission_offsets = self._collect_compare_submission_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=not use_confidence,
        )
        right_expert_offsets = self._collect_compare_expert_offsets(
            right_artifact,
            right_selected_submission_ids,
            use_confidence=use_confidence,
        )
        left_cat_features = {
            str(feature_name)
            for feature_name in (left_artifact.get("cat_features") or [])
        }
        right_cat_features = {
            str(feature_name)
            for feature_name in (right_artifact.get("cat_features") or [])
        }
        def score_from_contributions(
            artifact: Dict[str, Any],
            contributions: Optional[Dict[str, float]],
            included_features: Optional[set[str]] = None,
        ) -> Optional[float]:
            if contributions is None:
                return None
            return float(artifact.get("base_prediction_offset", 0.0) or 0.0) + sum(
                value
                for feature_name, value in contributions.items()
                if included_features is None or feature_name in included_features
            )

        # Calculate average contributions on the full-data reference grid.
        # The full-data artifact supplies the reference curve grid. Its base
        # curve is used without imported edits.
        original_uniform_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=left_artifact,
                evaluated_offsets={},
                method="uniform",
                numeric_grid_artifact=right_artifact,
            )
        )
        unconstrained_edited_uniform_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets=right_offsets,
                feature_shifts=right_manual_feature_shifts,
                method="uniform",
                numeric_grid_artifact=right_artifact,
            )
        )
        right_base_uniform_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets={},
                method="uniform",
                numeric_grid_artifact=right_artifact,
            )
        )
        original_average_prediction_score = score_from_contributions(
            left_artifact,
            original_uniform_contributions,
        )
        reduced_average_prediction_score = score_from_contributions(
            right_artifact,
            right_base_uniform_contributions,
        )
        mean_deviation_original_average = score_from_contributions(
            left_artifact,
            original_uniform_contributions,
            mean_deviation_feature_set,
        )
        unconstrained_mean_deviation_edited_average = score_from_contributions(
            right_artifact,
            unconstrained_edited_uniform_contributions,
            mean_deviation_feature_set,
        )
        right_base_mean_deviation_average = score_from_contributions(
            right_artifact,
            right_base_uniform_contributions,
            mean_deviation_feature_set,
        )
        unconstrained_scaled_feature_average = score_from_contributions(
            right_artifact,
            unconstrained_edited_uniform_contributions,
            mean_deviation_scaled_feature_set,
        )
        right_base_scaled_feature_average = score_from_contributions(
            right_artifact,
            right_base_uniform_contributions,
            mean_deviation_scaled_feature_set,
        )
        if (
            original_average_prediction_score is None
            or reduced_average_prediction_score is None
            or mean_deviation_original_average is None
            or unconstrained_mean_deviation_edited_average is None
            or right_base_mean_deviation_average is None
            or unconstrained_scaled_feature_average is None
            or right_base_scaled_feature_average is None
        ):
            raise ValueError("Unable to calculate average predictions on the full-data reference")
        mean_deviation_limit = 0.10 * abs(mean_deviation_original_average)
        scale_factor = 1.0
        right_effective_feature_shifts = dict(right_manual_feature_shifts)
        if (
            should_limit_edited_mean_deviation
            and abs(
                unconstrained_mean_deviation_edited_average
                - mean_deviation_original_average
            )
            > mean_deviation_limit
        ):
            lower_bound = mean_deviation_original_average - mean_deviation_limit
            upper_bound = mean_deviation_original_average + mean_deviation_limit
            target_average = min(
                max(unconstrained_mean_deviation_edited_average, lower_bound),
                upper_bound,
            )
            scaled_feature_edit_mean_effect = (
                unconstrained_scaled_feature_average
                - right_base_scaled_feature_average
            )
            if abs(scaled_feature_edit_mean_effect) > 1e-12:
                scale_factor = min(
                    1.0,
                    max(
                        0.0,
                        1.0
                        + (
                            target_average
                            - unconstrained_mean_deviation_edited_average
                        )
                        / scaled_feature_edit_mean_effect,
                    ),
                )
                right_offsets = self._scale_compare_offsets(
                    right_offsets,
                    scale_factor,
                    feature_names=mean_deviation_scaled_feature_set,
                )
                right_effective_feature_shifts = self._scale_compare_feature_shifts(
                    right_manual_feature_shifts,
                    scale_factor,
                    feature_names=mean_deviation_scaled_feature_set,
                )
                for feature_name, trace in right_submission_offsets.items():
                    if feature_name not in mean_deviation_scaled_feature_set:
                        continue
                    for submission_trace in trace:
                        submission_trace["offsets"] = {
                            value: float(offset) * scale_factor
                            for value, offset in submission_trace["offsets"].items()
                        }

        alternate_confidence_feature_shifts = dict(right_manual_feature_shifts)
        if should_limit_edited_mean_deviation:
            alternate_contributions = (
                self._compute_compare_feature_average_contributions_on_reference(
                    reference_artifact=left_artifact,
                    evaluated_artifact=right_artifact,
                    evaluated_offsets=alternate_confidence_offsets,
                    feature_shifts=right_manual_feature_shifts,
                    method="uniform",
                    numeric_grid_artifact=right_artifact,
                )
            )
            alternate_average = score_from_contributions(
                right_artifact,
                alternate_contributions,
                mean_deviation_feature_set,
            )
            alternate_scaled_average = score_from_contributions(
                right_artifact,
                alternate_contributions,
                mean_deviation_scaled_feature_set,
            )
            if alternate_average is None or alternate_scaled_average is None:
                raise ValueError("Unable to calculate alternate confidence average")
            if abs(alternate_average - mean_deviation_original_average) > mean_deviation_limit:
                lower_bound = mean_deviation_original_average - mean_deviation_limit
                upper_bound = mean_deviation_original_average + mean_deviation_limit
                target_average = min(max(alternate_average, lower_bound), upper_bound)
                scaled_edit_effect = (
                    alternate_scaled_average - right_base_scaled_feature_average
                )
                if abs(scaled_edit_effect) > 1e-12:
                    alternate_scale_factor = min(
                        1.0,
                        max(
                            0.0,
                            1.0 + (target_average - alternate_average) / scaled_edit_effect,
                        ),
                    )
                    alternate_confidence_offsets = self._scale_compare_offsets(
                        alternate_confidence_offsets,
                        alternate_scale_factor,
                        feature_names=mean_deviation_scaled_feature_set,
                    )
                    alternate_confidence_feature_shifts = self._scale_compare_feature_shifts(
                        right_manual_feature_shifts,
                        alternate_scale_factor,
                        feature_names=mean_deviation_scaled_feature_set,
                    )
        if apply_training_intercept_correction:
            training_rows = right_artifact.get("training_feature_rows") or []
            if not training_rows:
                raise ValueError(
                    "Training-only intercept correction requires reduced-model training rows"
                )
            original_training_predictions = self._predict_model_compare_artifact_rows(
                right_artifact, training_rows,
            )
            edited_training_predictions = self._predict_model_compare_artifact_rows(
                right_artifact, training_rows,
                offsets=right_offsets,
                feature_shifts=right_effective_feature_shifts,
            )
            intercept_adjustment = float(np.mean(
                original_training_predictions - edited_training_predictions
            ))
            adjusted_edited_base = original_edited_base + intercept_adjustment
        edited_uniform_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets=right_offsets,
                feature_shifts=right_effective_feature_shifts,
                method="uniform",
                numeric_grid_artifact=right_artifact,
            )
        )
        edited_average_prediction_score = score_from_contributions(
            right_artifact,
            edited_uniform_contributions,
        )
        constrained_mean_deviation_edited_average = score_from_contributions(
            right_artifact,
            edited_uniform_contributions,
            mean_deviation_feature_set,
        )
        if (
            edited_average_prediction_score is None
            or constrained_mean_deviation_edited_average is None
        ):
            raise ValueError("Unable to calculate edited average on the full-data reference")
        edited_average_prediction_score += intercept_adjustment
        mean_deviation_constraint = {
            "enabled": should_limit_edited_mean_deviation,
            "applied": bool(scale_factor < (1.0 - 1e-12)),
            "scale_factor": scale_factor,
            "limit": mean_deviation_limit,
            "original_average": mean_deviation_original_average,
            "unconstrained_edited_average": unconstrained_mean_deviation_edited_average,
            "constrained_edited_average": constrained_mean_deviation_edited_average,
            "satisfied": bool(
                abs(
                    constrained_mean_deviation_edited_average
                    - mean_deviation_original_average
                )
                <= mean_deviation_limit + 1e-9
            ),
            "feature_scale_factors": {
                feature_name: scale_factor
                for feature_name in explicitly_scaled_mean_deviation_feature_set
            },
        }
        original_test_set_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=left_artifact,
                evaluated_offsets={},
                method="test_set",
            )
        )
        edited_test_set_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets=right_offsets,
                feature_shifts=right_effective_feature_shifts,
                method="test_set",
            )
        )
        reduced_test_set_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets={},
                method="test_set",
            )
        )
        original_test_set_average_prediction_score = score_from_contributions(
            left_artifact,
            original_test_set_contributions,
        )
        reduced_test_set_average_prediction_score = score_from_contributions(
            right_artifact,
            reduced_test_set_contributions,
        )
        edited_test_set_average_prediction_score = score_from_contributions(
            right_artifact,
            edited_test_set_contributions,
        )
        if edited_test_set_average_prediction_score is not None:
            edited_test_set_average_prediction_score += intercept_adjustment

        missing_indicator_features = self._compare_missing_indicator_features(
            left_artifact, right_artifact
        )
        average_feature_names = [
            feature_name
            for feature_name in dict.fromkeys(
                (left_artifact.get("feature_names") or [])
                + (right_artifact.get("feature_names") or [])
            )
            if feature_name not in missing_indicator_features
        ]
        # The original artifact's exported 80% training partition is the
        # development reference, regardless of the currently active page model.
        main_dataset_rows = left_artifact.get("training_feature_rows") or []
        right_main_dataset_rows = (
            self._project_compare_rows(
                main_dataset_rows, left_artifact, right_artifact,
            ) if main_dataset_rows else []
        )
        original_main_dataset_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=left_artifact,
                evaluated_offsets={},
                method="main_dataset",
                sample_rows=main_dataset_rows,
            )
        )
        edited_main_dataset_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets=right_offsets,
                feature_shifts=right_effective_feature_shifts,
                method="main_dataset",
                sample_rows=right_main_dataset_rows,
            )
        )
        reduced_main_dataset_contributions = (
            self._compute_compare_feature_average_contributions_on_reference(
                reference_artifact=left_artifact,
                evaluated_artifact=right_artifact,
                evaluated_offsets={},
                method="main_dataset",
                sample_rows=right_main_dataset_rows,
            )
        )
        original_main_dataset_average_prediction_score = score_from_contributions(
            left_artifact, original_main_dataset_contributions
        )
        reduced_main_dataset_average_prediction_score = score_from_contributions(
            right_artifact, reduced_main_dataset_contributions
        )
        edited_main_dataset_average_prediction_score = score_from_contributions(
            right_artifact, edited_main_dataset_contributions
        )
        main_dataset_average_prediction_shift = (
            edited_main_dataset_average_prediction_score
            - reduced_main_dataset_average_prediction_score
            if edited_main_dataset_average_prediction_score is not None
            and reduced_main_dataset_average_prediction_score is not None
            else None
        )
        if edited_main_dataset_average_prediction_score is not None:
            edited_main_dataset_average_prediction_score += intercept_adjustment

        def build_contribution_rows(
            original_contributions: Optional[Dict[str, float]],
            reduced_contributions: Optional[Dict[str, float]],
            edited_contributions: Optional[Dict[str, float]],
            feature_names: Optional[List[str]] = None,
        ) -> Optional[List[Dict[str, Any]]]:
            if (
                original_contributions is None
                or reduced_contributions is None
                or edited_contributions is None
            ):
                return None

            original_base = float(left_artifact.get("base_prediction_offset", 0.0) or 0.0)
            reduced_base = original_edited_base
            edited_base = adjusted_edited_base
            rows: List[Dict[str, Any]] = [
                {
                    "feature_name": "__base_prediction_offset__",
                    "original": original_base,
                    "reduced": reduced_base,
                    "edited": edited_base,
                    "difference": edited_base - original_base,
                    "is_base_offset": True,
                }
            ]
            public_features = set(left_artifact.get("selected_feature_columns") or [])
            for feature_name in (
                feature_names or left_artifact.get("selected_feature_columns") or []
            ):
                if feature_name in missing_indicator_features:
                    continue
                if feature_names is None and (
                    feature_name not in original_contributions
                    or feature_name not in edited_contributions
                ):
                    continue
                original_value = float(original_contributions.get(feature_name, 0.0))
                reduced_value = float(reduced_contributions.get(feature_name, 0.0))
                edited_value = float(edited_contributions.get(feature_name, 0.0))
                rows.append(
                    {
                        "feature_name": str(feature_name),
                        "original": original_value,
                        "reduced": reduced_value,
                        "edited": edited_value,
                        "difference": edited_value - original_value,
                        "is_base_offset": False,
                        "is_internal_feature": feature_name not in public_features,
                    }
                )
            return rows

        # Build chart traces and per-feature metrics for the selected features.
        feature_previews = []
        overall_shape_function_mae_metrics = []
        expert_complete_model_mae_metrics = []
        for feature_name in active_features:
            left_shape = _json_safe(left_shape_map.get(feature_name) or {})
            right_shape = _json_safe(right_shape_map.get(feature_name) or {})
            if not left_shape or not right_shape:
                continue

            left_x_values = list(left_shape.get("x_values") or [])
            right_x_values = list(right_shape.get("x_values") or [])
            left_base_y_values = [float(value) for value in (left_shape.get("y_values") or [])]
            right_model_base_y_values = [
                float(value) for value in (right_shape.get("y_values") or [])
            ]
            if len(left_x_values) != len(left_base_y_values) or len(right_x_values) != len(right_model_base_y_values):
                continue

            right_base_y_values = list(right_model_base_y_values)

            left_effective_y_values = []
            for x_value, base_y in zip(left_x_values, left_base_y_values):
                offset = self._get_compare_offset_for_feature_value(
                    shape_function=left_shape,
                    original_is_categorical=feature_name in left_cat_features,
                    value=x_value,
                    offsets=left_offsets.get(feature_name, {}),
                )
                left_effective_y_values.append(float(base_y) + float(offset))

            right_effective_y_values = []
            right_alternate_confidence_y_values = []
            for x_value, base_y in zip(right_x_values, right_model_base_y_values):
                offset = self._get_compare_offset_for_feature_value(
                    shape_function=right_shape,
                    original_is_categorical=feature_name in right_cat_features,
                    value=x_value,
                    offsets=right_offsets.get(feature_name, {}),
                )
                right_effective_y_values.append(
                    float(base_y)
                    + float(offset)
                    + float(right_effective_feature_shifts.get(feature_name, 0.0))
                )
                alternate_offset = self._get_compare_offset_for_feature_value(
                    shape_function=right_shape,
                    original_is_categorical=feature_name in right_cat_features,
                    value=x_value,
                    offsets=alternate_confidence_offsets.get(feature_name, {}),
                )
                right_alternate_confidence_y_values.append(
                    float(base_y)
                    + float(alternate_offset)
                    + float(alternate_confidence_feature_shifts.get(feature_name, 0.0))
                )

            def build_individual_edit_traces(
                submission_offsets: List[Dict[str, Any]],
                *,
                shape_function: Dict[str, Any],
                is_categorical: bool,
                x_values: List[Any],
                base_y_values: List[float],
                feature_shift: float = 0.0,
            ) -> List[Dict[str, Any]]:
                return [
                    {
                        "submission_id": trace["submission_id"],
                        "user_name": _format_compare_chart_trace_user_name(
                            trace["user_name"]
                        ),
                        "sureness": trace["sureness"],
                        "y_values": [
                            float(base_y)
                            + self._get_compare_offset_for_feature_value(
                                shape_function=shape_function,
                                original_is_categorical=is_categorical,
                                value=x_value,
                                offsets=trace["offsets"],
                            )
                            + float(feature_shift)
                            for x_value, base_y in zip(x_values, base_y_values)
                        ],
                    }
                    for trace in submission_offsets
                ]

            left_individual_edit_traces = build_individual_edit_traces(
                left_submission_offsets.get(feature_name, []),
                shape_function=left_shape,
                is_categorical=feature_name in left_cat_features,
                x_values=left_x_values,
                base_y_values=left_base_y_values,
            )
            right_individual_edit_traces = build_individual_edit_traces(
                right_submission_offsets.get(feature_name, []),
                shape_function=right_shape,
                is_categorical=feature_name in right_cat_features,
                x_values=right_x_values,
                base_y_values=right_model_base_y_values,
                feature_shift=right_effective_feature_shifts.get(feature_name, 0.0),
            )

            feature_type = str(left_shape.get("feature_type", "numeric"))

            card_x_values = None
            card_original_values = None
            if feature_metrics_on_original_test_set:
                card_x_values = [row.get(feature_name) for row in original_test_rows]
                predict_shape = (
                    self._predict_categorical_shape_value
                    if feature_type == "categorical"
                    else self._predict_numeric_shape_value
                )
                card_original_values = [
                    predict_shape(left_shape, value) for value in card_x_values
                ]
                card_baseline_values = [
                    predict_shape(right_shape, value) for value in card_x_values
                ]
                card_edited_values = [
                    baseline_value
                    + self._get_compare_offset_for_feature_value(
                        shape_function=right_shape,
                        original_is_categorical=feature_name in right_cat_features,
                        value=value,
                        offsets=right_offsets.get(feature_name, {}),
                    )
                    + float(right_effective_feature_shifts.get(feature_name, 0.0))
                    for value, baseline_value in zip(
                        card_x_values, card_baseline_values
                    )
                ]
            elif feature_type == "numeric":
                left_numeric_x = np.asarray(left_x_values, dtype=float)
                right_numeric_x = np.asarray(right_x_values, dtype=float)
                overlap_min = max(float(np.min(left_numeric_x)), float(np.min(right_numeric_x)))
                overlap_max = min(float(np.max(left_numeric_x)), float(np.max(right_numeric_x)))
                if overlap_min > overlap_max:
                    raise ValueError(
                        f"Cannot compute per-feature metrics without overlapping numeric domain for '{feature_name}'"
                    )
                card_x_values = np.linspace(
                    overlap_min,
                    overlap_max,
                    MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS,
                ).tolist()
                card_original_values = self._interpolate_compare_values(
                    left_x_values, left_base_y_values, card_x_values
                )
                card_baseline_values = self._interpolate_compare_values(
                    right_x_values, right_model_base_y_values, card_x_values
                )
                card_edited_values = self._interpolate_compare_values(
                    right_x_values, right_effective_y_values, card_x_values
                )

            def calculate_original_base_mae(
                trace_x_values: List[Any],
                trace_y_values: List[float],
                *,
                trace_shape: Dict[str, Any],
                trace_offsets: Dict[Any, float],
                trace_is_categorical: bool,
                trace_feature_shift: float = 0.0,
            ) -> float:
                if feature_metrics_on_original_test_set:
                    sampled_trace_values = [
                        predict_shape(trace_shape, value)
                        + self._get_compare_offset_for_feature_value(
                            shape_function=trace_shape,
                            original_is_categorical=trace_is_categorical,
                            value=value,
                            offsets=trace_offsets,
                        )
                        + trace_feature_shift
                        for value in card_x_values
                    ]
                    return self._compute_compare_mae(
                        card_original_values, sampled_trace_values
                    )
                if feature_type == "numeric":
                    sampled_trace_values = self._interpolate_compare_values(
                        trace_x_values, trace_y_values, card_x_values
                    )
                    return self._compute_compare_mae(
                        card_original_values, sampled_trace_values
                    )
                return self._compute_compare_mae(left_base_y_values, trace_y_values)

            for trace, submission in zip(
                left_individual_edit_traces,
                left_submission_offsets.get(feature_name, []),
            ):
                trace["original_base_mae"] = calculate_original_base_mae(
                    left_x_values,
                    trace["y_values"],
                    trace_shape=left_shape,
                    trace_offsets=submission["offsets"],
                    trace_is_categorical=feature_name in left_cat_features,
                )
            alternate_submissions = {
                submission["submission_id"]: submission
                for submission in right_alternate_submission_offsets.get(feature_name, [])
            }
            for trace, submission in zip(
                right_individual_edit_traces,
                right_submission_offsets.get(feature_name, []),
            ):
                trace["original_base_mae"] = calculate_original_base_mae(
                    right_x_values,
                    trace["y_values"],
                    trace_shape=right_shape,
                    trace_offsets=submission["offsets"],
                    trace_is_categorical=feature_name in right_cat_features,
                    trace_feature_shift=float(
                        right_effective_feature_shifts.get(feature_name, 0.0)
                    ),
                )
                alternate_submission = alternate_submissions.get(trace["submission_id"])
                if alternate_submission is None:
                    continue
                alternate_y_values = [
                    float(base_y)
                    + self._get_compare_offset_for_feature_value(
                        shape_function=right_shape,
                        original_is_categorical=feature_name in right_cat_features,
                        value=x_value,
                        offsets=alternate_submission["offsets"],
                    )
                    + float(alternate_confidence_feature_shifts.get(feature_name, 0.0))
                    for x_value, base_y in zip(right_x_values, right_model_base_y_values)
                ]
                alternate_mae = calculate_original_base_mae(
                    right_x_values,
                    alternate_y_values,
                    trace_shape=right_shape,
                    trace_offsets=alternate_submission["offsets"],
                    trace_is_categorical=feature_name in right_cat_features,
                    trace_feature_shift=float(
                        alternate_confidence_feature_shifts.get(feature_name, 0.0)
                    ),
                )
                trace["original_base_mae_with_confidence"] = (
                    trace["original_base_mae"] if use_confidence else alternate_mae
                )
                trace["original_base_mae_without_confidence"] = (
                    alternate_mae if use_confidence else trace["original_base_mae"]
                )

            baseline_edited_vs_edited_base_mae = self._compute_compare_mae(
                right_model_base_y_values,
                right_model_base_y_values,
            )
            baseline_edited_vs_edited_base_rmse = self._compute_compare_rmse(
                right_model_base_y_values,
                right_model_base_y_values,
            )
            baseline_edited_vs_edited_base_r2 = self._compute_compare_r2(
                right_model_base_y_values,
                right_model_base_y_values,
            )
            edited_vs_edited_base_mae = self._compute_compare_mae(
                right_model_base_y_values,
                right_effective_y_values,
            )
            edited_vs_edited_base_rmse = self._compute_compare_rmse(
                right_model_base_y_values,
                right_effective_y_values,
            )
            edited_vs_edited_base_r2 = self._compute_compare_r2(
                right_model_base_y_values,
                right_effective_y_values,
            )
            if feature_type == "numeric":
                baseline_edited_vs_original_base_mae = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_model_base_y_values,
                        metric_name="MAE",
                    )
                )
                baseline_edited_vs_original_base_rmse = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_model_base_y_values,
                        metric_name="RMSE",
                    )
                )
                baseline_edited_vs_original_base_r2 = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_model_base_y_values,
                        metric_name="R2",
                    )
                )
                edited_vs_original_base_mae = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_effective_y_values,
                        metric_name="MAE",
                    )
                )
                edited_vs_original_base_rmse = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_effective_y_values,
                        metric_name="RMSE",
                    )
                )
                edited_vs_original_base_r2 = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=right_effective_y_values,
                        metric_name="R2",
                    )
                )
            else:
                baseline_edited_vs_original_base_mae = self._compute_compare_mae(
                    left_base_y_values,
                    right_model_base_y_values,
                )
                baseline_edited_vs_original_base_rmse = self._compute_compare_rmse(
                    left_base_y_values,
                    right_model_base_y_values,
                )
                baseline_edited_vs_original_base_r2 = self._compute_compare_r2(
                    left_base_y_values,
                    right_model_base_y_values,
                )
                edited_vs_original_base_mae = self._compute_compare_mae(
                    left_base_y_values,
                    right_effective_y_values,
                )
                edited_vs_original_base_rmse = self._compute_compare_rmse(
                    left_base_y_values,
                    right_effective_y_values,
                )
                edited_vs_original_base_r2 = self._compute_compare_r2(
                    left_base_y_values,
                    right_effective_y_values,
                )

            if feature_type == "numeric":
                right_numeric_x_values = np.asarray(right_x_values, dtype=float)
                left_numeric_x_values = np.asarray(left_x_values, dtype=float)
                comparison_point_count = int(
                    np.count_nonzero(
                        (left_numeric_x_values >= (np.min(right_numeric_x_values) - 1e-9))
                        & (left_numeric_x_values <= (np.max(right_numeric_x_values) + 1e-9))
                    )
                )
            else:
                comparison_point_count = len(left_base_y_values)
            if feature_name in overall_mae_feature_set:
                overall_shape_function_mae_metrics.append(
                    {
                        "point_count": comparison_point_count,
                        "baseline": baseline_edited_vs_original_base_mae,
                        "edited": edited_vs_original_base_mae,
                    }
                )
            for expert in right_expert_offsets.values():
                expert_y_values = [
                    float(base_y)
                    + self._get_compare_offset_for_feature_value(
                        shape_function=right_shape,
                        original_is_categorical=feature_name in right_cat_features,
                        value=x_value,
                        offsets=expert["offsets"].get(feature_name, {}),
                    )
                    for x_value, base_y in zip(
                        right_x_values,
                        right_model_base_y_values,
                    )
                ]
                expert_mae = (
                    self._compute_numeric_compare_metric_on_reference_grid(
                        reference_x_values=left_x_values,
                        reference_y_values=left_base_y_values,
                        comparison_x_values=right_x_values,
                        comparison_y_values=expert_y_values,
                        metric_name="MAE",
                    )
                    if feature_type == "numeric"
                    else self._compute_compare_mae(
                        left_base_y_values,
                        expert_y_values,
                    )
                )
                expert_complete_model_mae_metrics.append(
                    {
                            "user_name": expert["user_name"],
                            "mae": expert_mae,
                            "baseline_mae": baseline_edited_vs_original_base_mae,
                            "point_count": comparison_point_count,
                        "submission_ids": expert["submission_ids"],
                    }
                )

            card_metrics = {}
            if card_original_values is not None:
                for metric_name, compute_metric, compare_status in (
                    ("mae", self._compute_compare_mae, self._compare_mae_status),
                    ("rmse", self._compute_compare_rmse, self._compare_mae_status),
                    ("r2", self._compute_compare_r2, self._compare_r2_status),
                ):
                    baseline_metric = compute_metric(card_original_values, card_baseline_values)
                    edited_metric = compute_metric(card_original_values, card_edited_values)
                    status_key = (
                        "edited_vs_original_base_status"
                        if metric_name == "mae"
                        else f"edited_vs_original_base_{metric_name}_status"
                    )
                    card_metrics.update(
                        {
                            f"baseline_edited_vs_original_base_{metric_name}": baseline_metric,
                            f"edited_vs_original_base_{metric_name}": edited_metric,
                            status_key: compare_status(edited_metric, baseline_metric),
                        }
                    )

            feature_previews.append(
                {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "left_x_values": left_x_values,
                    "left_x_tick_labels": left_shape.get("x_tick_labels"),
                    "right_x_values": right_x_values,
                    "right_x_tick_labels": right_shape.get("x_tick_labels"),
                    "left_base_y_values": left_base_y_values,
                    "left_effective_y_values": left_effective_y_values,
                    "right_base_y_values": right_base_y_values,
                    "right_effective_y_values": right_effective_y_values,
                    "right_alternate_confidence_y_values": right_alternate_confidence_y_values,
                    "left_individual_edit_traces": left_individual_edit_traces,
                    "right_individual_edit_traces": right_individual_edit_traces,
                    "left_chart_config": left_shape.get("chart_config") or {},
                    "right_chart_config": right_shape.get("chart_config") or {},
                    "comparison_metrics": {
                        "edited_vs_edited_base_mae": edited_vs_edited_base_mae,
                        "edited_vs_original_base_mae": edited_vs_original_base_mae,
                        "baseline_edited_vs_edited_base_mae": baseline_edited_vs_edited_base_mae,
                        "baseline_edited_vs_original_base_mae": baseline_edited_vs_original_base_mae,
                        "edited_vs_edited_base_status": self._compare_mae_status(
                            edited_vs_edited_base_mae,
                            baseline_edited_vs_edited_base_mae,
                        ),
                        "edited_vs_original_base_status": self._compare_mae_status(
                            edited_vs_original_base_mae,
                            baseline_edited_vs_original_base_mae,
                        ),
                        "edited_vs_edited_base_rmse": edited_vs_edited_base_rmse,
                        "edited_vs_original_base_rmse": edited_vs_original_base_rmse,
                        "baseline_edited_vs_edited_base_rmse": baseline_edited_vs_edited_base_rmse,
                        "baseline_edited_vs_original_base_rmse": baseline_edited_vs_original_base_rmse,
                        "edited_vs_edited_base_rmse_status": self._compare_mae_status(
                            edited_vs_edited_base_rmse,
                            baseline_edited_vs_edited_base_rmse,
                        ),
                        "edited_vs_original_base_rmse_status": self._compare_mae_status(
                            edited_vs_original_base_rmse,
                            baseline_edited_vs_original_base_rmse,
                        ),
                        "edited_vs_edited_base_r2": edited_vs_edited_base_r2,
                        "edited_vs_original_base_r2": edited_vs_original_base_r2,
                        "baseline_edited_vs_edited_base_r2": baseline_edited_vs_edited_base_r2,
                        "baseline_edited_vs_original_base_r2": baseline_edited_vs_original_base_r2,
                        "edited_vs_edited_base_r2_status": self._compare_r2_status(
                            edited_vs_edited_base_r2,
                            baseline_edited_vs_edited_base_r2,
                        ),
                        "edited_vs_original_base_r2_status": self._compare_r2_status(
                            edited_vs_original_base_r2,
                            baseline_edited_vs_original_base_r2,
                        ),
                        **card_metrics,
                    },
                }
            )

        # Summarize held-out and training performance after applying edits.
        main_training_performance_inputs = self._main_training_performance_inputs(
            left_artifact, right_artifact
        )
        if held_out_test_source == "main_training":
            if main_training_performance_inputs is not None:
                training_feature_names, training_targets = (
                    main_training_performance_inputs
                )
                main_training_performance_data = {
                    "feature_rows": _json_safe(
                        self.X_train.loc[:, training_feature_names].to_dict(
                            orient="records"
                        )
                    ),
                    "target_values": [float(value) for value in training_targets],
                }
            else:
                main_training_performance_data = None
            performance_data_artifact = {
                "held_out_test_data": main_training_performance_data,
                "missing_indicator_map": left_artifact.get("missing_indicator_map") or {},
            }
        else:
            performance_data_artifact = (
                left_artifact if held_out_test_source == "left" else right_artifact
            )
        held_out_test_performance = self._build_held_out_test_performance(
            original_artifact=left_artifact,
            edited_artifact=right_artifact,
            test_data_artifact=performance_data_artifact,
            test_data_source=held_out_test_source,
            edited_offsets=right_offsets,
            edited_feature_shifts=right_effective_feature_shifts,
            intercept_adjustment=intercept_adjustment,
            intercept_correction_source=(
                "reduced training rows" if apply_training_intercept_correction
                else "manual"
            ),
        )

        training_mean_prediction_summary = None
        training_rows = right_artifact.get("training_feature_rows") or []
        if training_rows:
            reduced_model_mean = float(np.mean(
                self._predict_model_compare_artifact_rows(
                    right_artifact,
                    training_rows,
                    exclude_missing_indicators=True,
                )
            ))
            edited_before_correction_mean = float(np.mean(
                self._predict_model_compare_artifact_rows(
                    right_artifact,
                    training_rows,
                    offsets=right_offsets,
                    feature_shifts=right_effective_feature_shifts,
                    exclude_missing_indicators=True,
                )
            ))
            difference_before_correction = (
                edited_before_correction_mean - reduced_model_mean
            )
            suggested_matching_adjustment = -difference_before_correction
            training_mean_prediction_summary = {
                "training_row_count": len(training_rows),
                "reduced_model_mean": reduced_model_mean,
                "edited_before_correction_mean": edited_before_correction_mean,
                "difference_before_correction": difference_before_correction,
                "edited_after_correction_mean": (
                    edited_before_correction_mean + intercept_adjustment
                ),
                "original_full_data_intercept": float(
                    left_artifact.get("base_prediction_offset", 0.0) or 0.0
                ),
                "reduced_intercept": original_edited_base,
                "edited_intercept_before_correction": original_edited_base,
                "intercept_adjustment": intercept_adjustment,
                "edited_intercept_after_correction": adjusted_edited_base,
                "suggested_matching_adjustment": suggested_matching_adjustment,
                "suggested_matching_intercept": (
                    original_edited_base + suggested_matching_adjustment
                ),
            }

        return {
            "use_confidence": bool(use_confidence),
            "intercept_adjustment": intercept_adjustment,
            "original_edited_base_prediction_offset": original_edited_base,
            "adjusted_edited_base_prediction_offset": adjusted_edited_base,
            "feature_previews": feature_previews,
            "all_feature_mae_change": self._build_all_feature_mae_change(
                left_artifact=left_artifact,
                right_artifact=right_artifact,
                shared_features=shared_features,
            ),
            "overall_shape_function_mae": {
                **self._build_overall_shape_function_mae(
                    overall_shape_function_mae_metrics,
                ),
                "expert_complete_model_mae": self._build_expert_complete_model_mae(
                    expert_complete_model_mae_metrics,
                ),
            },
            "held_out_test_performance": held_out_test_performance,
            "main_training_performance_available": bool(
                main_training_performance_inputs
            ),
            "average_prediction_scores": {
                "original": original_average_prediction_score,
                "reduced": reduced_average_prediction_score,
                "edited": edited_average_prediction_score,
                "difference": (
                    edited_average_prediction_score - original_average_prediction_score
                ),
                "edited_minus_reduced": (
                    edited_average_prediction_score - reduced_average_prediction_score
                ),
            },
            "training_mean_prediction_summary": training_mean_prediction_summary,
            "test_set_average_prediction_scores": (
                {
                    "original": original_test_set_average_prediction_score,
                    "reduced": reduced_test_set_average_prediction_score,
                    "edited": edited_test_set_average_prediction_score,
                    "difference": (
                        edited_test_set_average_prediction_score
                        - original_test_set_average_prediction_score
                    ),
                    "edited_minus_reduced": (
                        edited_test_set_average_prediction_score
                        - reduced_test_set_average_prediction_score
                    ),
                }
                if original_test_set_average_prediction_score is not None
                and reduced_test_set_average_prediction_score is not None
                and edited_test_set_average_prediction_score is not None
                else None
            ),
            "main_dataset_average_prediction_scores": (
                {
                    "original": original_main_dataset_average_prediction_score,
                    "reduced": reduced_main_dataset_average_prediction_score,
                    "edited": edited_main_dataset_average_prediction_score,
                    "difference": (
                        edited_main_dataset_average_prediction_score
                        - original_main_dataset_average_prediction_score
                    ),
                    "edited_minus_reduced": (
                        edited_main_dataset_average_prediction_score
                        - reduced_main_dataset_average_prediction_score
                    ),
                }
                if original_main_dataset_average_prediction_score is not None
                and reduced_main_dataset_average_prediction_score is not None
                and edited_main_dataset_average_prediction_score is not None
                else None
            ),
            "main_dataset_training_row_count": len(main_dataset_rows),
            "main_dataset_average_prediction_shift": (
                main_dataset_average_prediction_shift
            ),
            "average_prediction_feature_contributions": {
                "uniform": build_contribution_rows(
                    original_uniform_contributions,
                    right_base_uniform_contributions,
                    edited_uniform_contributions,
                ),
                "test_set": build_contribution_rows(
                    original_test_set_contributions,
                    reduced_test_set_contributions,
                    edited_test_set_contributions,
                    average_feature_names,
                ),
                "main_dataset": build_contribution_rows(
                    original_main_dataset_contributions,
                    reduced_main_dataset_contributions,
                    edited_main_dataset_contributions,
                    average_feature_names,
                ),
            },
            "mean_deviation_constraint": mean_deviation_constraint,
        }
