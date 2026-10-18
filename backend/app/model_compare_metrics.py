"""Metric and contribution calculations for exported model comparisons."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
import numpy as np

from .data_processing import MISSING_INDICATOR_SUFFIX
from .ml_shared import MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS


class ModelCompareMetricsMixin:
    def _main_training_performance_inputs(
        self,
        original_artifact: Dict[str, Any],
        edited_artifact: Dict[str, Any],
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """Check the active main-page training split for debug evaluation."""
        if (
            not self.is_trained
            or self.model_source != "trained"
            or self.X_train is None
            or self.X_train.empty
            or self.y_train is None
        ):
            return None
        feature_names = list(dict.fromkeys(
            (original_artifact.get("feature_names") or [])
            + (edited_artifact.get("feature_names") or [])
        ))
        if not set(feature_names).issubset(self.X_train.columns):
            return None
        targets = np.asarray(self.y_train, dtype=float).reshape(-1)
        if len(targets) != len(self.X_train) or not np.all(np.isfinite(targets)):
            return None
        return feature_names, targets

    @staticmethod
    def _compute_compare_mae(
        baseline_y_values: List[float],
        effective_y_values: List[float],
    ) -> float:
        if len(baseline_y_values) != len(effective_y_values):
            raise ValueError("Cannot compute compare MAE for misaligned feature values")
        if not baseline_y_values:
            return 0.0
        return float(mean_absolute_error(baseline_y_values, effective_y_values))

    @staticmethod
    def _compute_compare_rmse(
        baseline_y_values: List[float],
        effective_y_values: List[float],
    ) -> float:
        if len(baseline_y_values) != len(effective_y_values):
            raise ValueError("Cannot compute compare RMSE for misaligned feature values")
        if not baseline_y_values:
            return 0.0
        return float(root_mean_squared_error(baseline_y_values, effective_y_values))

    @staticmethod
    def _compute_compare_r2(
        baseline_y_values: List[float],
        effective_y_values: List[float],
    ) -> float:
        if len(baseline_y_values) != len(effective_y_values):
            raise ValueError("Cannot compute compare R2 for misaligned feature values")
        if len(baseline_y_values) < 2:
            return 1.0
        return float(r2_score(baseline_y_values, effective_y_values))

    @staticmethod
    def _interpolate_compare_values(
        source_x_values: List[Any],
        source_y_values: List[float],
        target_x_values: List[Any],
    ) -> List[float]:
        if (
            len(source_x_values) != len(source_y_values)
            or not source_x_values
            or not target_x_values
        ):
            return []

        try:
            source_x = np.asarray(source_x_values, dtype=float)
            source_y = np.asarray(source_y_values, dtype=float)
            target_x = np.asarray(target_x_values, dtype=float)
        except (TypeError, ValueError):
            return []

        if (
            source_x.size == 0
            or source_y.size == 0
            or source_x.size != source_y.size
            or target_x.size == 0
        ):
            return []

        order = np.argsort(source_x)
        source_x = source_x[order]
        source_y = source_y[order]

        return [
            float(value)
            for value in np.interp(target_x, source_x, source_y).tolist()
        ]

    @classmethod
    def _compute_numeric_compare_mae_on_reference_grid(
        cls,
        *,
        reference_x_values: List[Any],
        reference_y_values: List[float],
        comparison_x_values: List[Any],
        comparison_y_values: List[float],
    ) -> float:
        if (
            len(reference_x_values) != len(reference_y_values)
            or len(comparison_x_values) != len(comparison_y_values)
            or not reference_x_values
            or not comparison_x_values
        ):
            raise ValueError("Cannot compute compare MAE for invalid numeric feature values")

        try:
            comparison_numeric_x = np.asarray(comparison_x_values, dtype=float)
            reference_numeric_x = np.asarray(reference_x_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Cannot compute compare MAE for non-numeric reference grid values"
            ) from exc

        if comparison_numeric_x.size == 0 or reference_numeric_x.size == 0:
            raise ValueError("Cannot compute compare MAE without numeric feature values")

        domain_min = float(np.min(comparison_numeric_x))
        domain_max = float(np.max(comparison_numeric_x))
        overlap_mask = (
            (reference_numeric_x >= (domain_min - 1e-9))
            & (reference_numeric_x <= (domain_max + 1e-9))
        )
        if not np.any(overlap_mask):
            raise ValueError("Cannot compute compare MAE without overlapping numeric domain")

        filtered_reference_x = reference_numeric_x[overlap_mask].tolist()
        filtered_reference_y = [
            float(value)
            for include, value in zip(overlap_mask.tolist(), reference_y_values)
            if include
        ]
        interpolated_comparison_y = cls._interpolate_compare_values(
            comparison_x_values,
            comparison_y_values,
            filtered_reference_x,
        )
        return cls._compute_compare_mae(
            filtered_reference_y,
            interpolated_comparison_y,
        )

    @classmethod
    def _compute_numeric_compare_metric_on_reference_grid(
        cls,
        *,
        reference_x_values: List[Any],
        reference_y_values: List[float],
        comparison_x_values: List[Any],
        comparison_y_values: List[float],
        metric_name: str,
    ) -> float:
        if (
            len(reference_x_values) != len(reference_y_values)
            or len(comparison_x_values) != len(comparison_y_values)
            or not reference_x_values
            or not comparison_x_values
        ):
            raise ValueError(
                f"Cannot compute compare {metric_name} for invalid numeric feature values"
            )

        try:
            comparison_numeric_x = np.asarray(comparison_x_values, dtype=float)
            reference_numeric_x = np.asarray(reference_x_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot compute compare {metric_name} for non-numeric reference grid values"
            ) from exc

        if comparison_numeric_x.size == 0 or reference_numeric_x.size == 0:
            raise ValueError(
                f"Cannot compute compare {metric_name} without numeric feature values"
            )

        domain_min = float(np.min(comparison_numeric_x))
        domain_max = float(np.max(comparison_numeric_x))
        overlap_mask = (
            (reference_numeric_x >= (domain_min - 1e-9))
            & (reference_numeric_x <= (domain_max + 1e-9))
        )
        if not np.any(overlap_mask):
            raise ValueError(
                f"Cannot compute compare {metric_name} without overlapping numeric domain"
            )

        filtered_reference_x = reference_numeric_x[overlap_mask].tolist()
        filtered_reference_y = [
            float(value)
            for include, value in zip(overlap_mask.tolist(), reference_y_values)
            if include
        ]
        interpolated_comparison_y = cls._interpolate_compare_values(
            comparison_x_values,
            comparison_y_values,
            filtered_reference_x,
        )
        if metric_name == "MAE":
            return cls._compute_compare_mae(
                filtered_reference_y,
                interpolated_comparison_y,
            )
        if metric_name == "RMSE":
            return cls._compute_compare_rmse(
                filtered_reference_y,
                interpolated_comparison_y,
            )
        if metric_name == "R2":
            return cls._compute_compare_r2(
                filtered_reference_y,
                interpolated_comparison_y,
            )
        raise ValueError(f"Unsupported compare metric '{metric_name}'")

    @staticmethod
    def _compare_mae_status(
        current_mae: float,
        baseline_mae: float,
    ) -> str:
        if abs(current_mae - baseline_mae) < 0.000001:
            return "unchanged"
        if current_mae < baseline_mae:
            return "decreased"
        return "increased"

    @staticmethod
    def _compare_r2_status(
        current_r2: float,
        baseline_r2: float,
    ) -> str:
        if abs(current_r2 - baseline_r2) < 0.000001:
            return "unchanged"
        if current_r2 > baseline_r2:
            return "increased"
        return "decreased"

    @staticmethod
    def _collect_compare_offsets(
        prepared_artifact: Dict[str, Any],
        selected_submission_ids: Optional[List[str]],
        *,
        use_confidence: bool,
    ) -> Dict[str, Dict[Any, float]]:
        selected_set = (
            {str(submission_id) for submission_id in selected_submission_ids}
            if selected_submission_ids is not None
            else None
        )
        aggregate: Dict[str, Dict[Any, Dict[str, float]]] = {}

        for feature_group in prepared_artifact.get("submissions_by_feature") or []:
            feature_name = str(feature_group.get("feature_name", "")).strip()
            if not feature_name:
                continue
            feature_bucket = aggregate.setdefault(feature_name, {})
            for submission in feature_group.get("submissions") or []:
                submission_id = str(submission.get("submission_id", "")).strip()
                if selected_set is not None and submission_id not in selected_set:
                    continue
                for point in submission.get("edited_points") or []:
                    raw_x_value = point.get("x_value")
                    point_bucket = feature_bucket.setdefault(
                        raw_x_value,
                        {"offset_sum": 0.0, "point_count": 0.0},
                    )
                    contribution = float(point.get("y_value", 0.0))
                    if use_confidence:
                        contribution *= float(point.get("weight", 0.5))
                    point_bucket["offset_sum"] += contribution
                    point_bucket["point_count"] += 1.0

        combined: Dict[str, Dict[Any, float]] = {}
        for feature_name, point_map in aggregate.items():
            combined[feature_name] = {}
            for raw_x_value, stats in point_map.items():
                point_count = float(stats.get("point_count", 0.0) or 0.0)
                combined[feature_name][raw_x_value] = (
                    float(stats.get("offset_sum", 0.0)) / point_count
                    if point_count > 0
                    else 0.0
                )
        return combined

    @staticmethod
    def _collect_compare_submission_offsets(
        prepared_artifact: Dict[str, Any],
        selected_submission_ids: Optional[List[str]],
        *,
        use_confidence: bool,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return one offset map per selected submission, grouped by feature.

        The aggregate preview averages all selected point contributions.  The
        individual-trace preview deliberately keeps each submission separate,
        while applying the same confidence weighting to its points.
        """
        selected_set = (
            {str(submission_id) for submission_id in selected_submission_ids}
            if selected_submission_ids is not None
            else None
        )
        traces: Dict[str, List[Dict[str, Any]]] = {}

        for feature_group in prepared_artifact.get("submissions_by_feature") or []:
            feature_name = str(feature_group.get("feature_name", "")).strip()
            if not feature_name:
                continue
            for submission in feature_group.get("submissions") or []:
                submission_id = str(submission.get("submission_id", "")).strip()
                if selected_set is not None and submission_id not in selected_set:
                    continue

                point_stats: Dict[Any, Dict[str, float]] = {}
                for point in submission.get("edited_points") or []:
                    raw_x_value = point.get("x_value")
                    stats = point_stats.setdefault(
                        raw_x_value,
                        {"offset_sum": 0.0, "point_count": 0.0},
                    )
                    contribution = float(point.get("y_value", 0.0))
                    if use_confidence:
                        contribution *= float(point.get("weight", 0.5))
                    stats["offset_sum"] += contribution
                    stats["point_count"] += 1.0

                offsets = {
                    raw_x_value: float(stats["offset_sum"])
                    / float(stats["point_count"])
                    for raw_x_value, stats in point_stats.items()
                    if float(stats["point_count"]) > 0
                }
                traces.setdefault(feature_name, []).append(
                    {
                        "submission_id": submission_id,
                        "user_name": str(submission.get("user_name", "") or ""),
                        "sureness": int(submission.get("sureness", 0) or 0),
                        "offsets": offsets,
                    }
                )
        return traces

    @staticmethod
    def _collect_compare_expert_offsets(
        prepared_artifact: Dict[str, Any],
        selected_submission_ids: Optional[List[str]],
        *,
        use_confidence: bool,
    ) -> Dict[str, Dict[str, Any]]:
        """Combine each selected expert's edits across all of their features."""
        selected_set = (
            {str(submission_id) for submission_id in selected_submission_ids}
            if selected_submission_ids is not None
            else None
        )
        experts: Dict[str, Dict[str, Any]] = {}

        for feature_group in prepared_artifact.get("submissions_by_feature") or []:
            feature_name = str(feature_group.get("feature_name", "")).strip()
            if not feature_name:
                continue
            for submission in feature_group.get("submissions") or []:
                submission_id = str(submission.get("submission_id", "")).strip()
                if selected_set is not None and submission_id not in selected_set:
                    continue
                user_name = str(submission.get("user_name", "") or "").strip()
                expert_key = user_name or submission_id
                if not expert_key:
                    continue
                expert = experts.setdefault(
                    expert_key,
                    {
                        "user_name": user_name or submission_id,
                        "submission_ids": set(),
                        "offset_stats": {},
                    },
                )
                expert["submission_ids"].add(submission_id)
                point_stats = expert["offset_stats"].setdefault(feature_name, {})
                for point in submission.get("edited_points") or []:
                    raw_x_value = point.get("x_value")
                    stats = point_stats.setdefault(
                        raw_x_value,
                        {"offset_sum": 0.0, "point_count": 0.0},
                    )
                    contribution = float(point.get("y_value", 0.0))
                    if use_confidence:
                        contribution *= float(point.get("weight", 0.5))
                    stats["offset_sum"] += contribution
                    stats["point_count"] += 1.0

        for expert in experts.values():
            expert["offsets"] = {
                feature_name: {
                    raw_x_value: float(stats["offset_sum"])
                    / float(stats["point_count"])
                    for raw_x_value, stats in point_stats.items()
                    if float(stats["point_count"]) > 0
                }
                for feature_name, point_stats in expert["offset_stats"].items()
            }
        return experts

    @classmethod
    def _compute_compare_average_prediction_score(
        cls,
        prepared_artifact: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
    ) -> float:
        """Estimate an artifact's mean prediction across its displayed grids.

        Each feature curve is sampled uniformly over the same x-values shown on
        the comparison page. Their means are additive in the exported IGANN
        representation, together with the artifact's base prediction offset.
        """
        contributions = cls._compute_compare_feature_average_contributions(
            prepared_artifact,
            offsets,
            method="uniform",
        )
        return float(prepared_artifact.get("base_prediction_offset", 0.0) or 0.0) + sum(
            contributions.values()
        )

    @classmethod
    def _compute_compare_data_weighted_average_prediction_score(
        cls,
        prepared_artifact: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
    ) -> Optional[float]:
        """Estimate an average using the artifact's feature distributions.

        Unlike the uniform measure, each contribution is weighted by the
        observed frequency of that feature value (or numeric bin). This is a
        marginal shape-function estimate, not a reconstruction of joint rows.
        """
        contributions = cls._compute_compare_feature_average_contributions(
            prepared_artifact,
            offsets,
            method="data_weighted",
        )
        if contributions is None:
            return None
        return float(prepared_artifact.get("base_prediction_offset", 0.0) or 0.0) + sum(
            contributions.values()
        )

    @staticmethod
    def _compare_missing_indicator_features(*artifacts: Dict[str, Any]) -> set[str]:
        """Identify generated missingness features in compared artifacts."""
        excluded = set()
        for artifact in artifacts:
            excluded.update(
                str(name)
                for name in (artifact.get("missing_indicator_map") or {}).values()
            )
            excluded.update(
                str(name)
                for name in (artifact.get("feature_names") or [])
                if str(name).endswith(MISSING_INDICATOR_SUFFIX)
            )
        return excluded

    @classmethod
    def _compute_compare_feature_average_contributions(
        cls,
        prepared_artifact: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
        *,
        method: str,
    ) -> Optional[Dict[str, float]]:
        """Return one averaging-method contribution per public shape function."""
        categorical_features = {
            str(feature_name)
            for feature_name in (prepared_artifact.get("cat_features") or [])
        }
        contributions: Dict[str, float] = {}

        for shape_function in prepared_artifact.get("shape_functions") or []:
            feature_name = str(shape_function.get("feature_name", "")).strip()
            if not feature_name:
                continue

            if method == "uniform":
                x_values = list(shape_function.get("x_values") or [])
                y_values = shape_function.get("y_values") or []
                if len(x_values) != len(y_values) or not x_values:
                    continue
                weighted_values = [(value, 1) for value in x_values]
            elif method == "data_weighted":
                distribution = shape_function.get("distribution") or {}
                if not distribution:
                    return None
                if distribution.get("chart_type") == "categorical":
                    weighted_values = [
                        (item.get("x_value"), int(item.get("count", 0) or 0))
                        for item in (distribution.get("counts") or [])
                        if int(item.get("count", 0) or 0) > 0
                    ]
                elif distribution.get("chart_type") == "numeric":
                    weighted_values = [
                        (item.get("center"), int(item.get("count", 0) or 0))
                        for item in (distribution.get("bins") or [])
                        if int(item.get("count", 0) or 0) > 0
                    ]
                else:
                    return None
            else:
                raise ValueError(f"Unknown average contribution method '{method}'")

            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight <= 0:
                return None

            contribution = 0.0
            for value, weight in weighted_values:
                if shape_function.get("feature_type") == "categorical":
                    base_value = cls._predict_categorical_shape_value(
                        shape_function,
                        value,
                    )
                else:
                    base_value = cls._predict_numeric_shape_value(
                        shape_function,
                        value,
                    )
                offset = cls._get_compare_offset_for_feature_value(
                    shape_function=shape_function,
                    original_is_categorical=feature_name in categorical_features,
                    value=value,
                    offsets=offsets.get(feature_name, {}),
                )
                contribution += weight * (base_value + float(offset))
            contributions[feature_name] = contribution / total_weight

        return contributions

    @classmethod
    def _compute_compare_feature_average_contributions_on_reference(
        cls,
        *,
        reference_artifact: Dict[str, Any],
        evaluated_artifact: Dict[str, Any],
        evaluated_offsets: Dict[str, Dict[Any, float]],
        feature_shifts: Optional[Dict[str, float]] = None,
        method: str,
        numeric_grid_artifact: Optional[Dict[str, Any]] = None,
        sample_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, float]]:
        """Evaluate both artifacts on the original artifact's feature support.

        This supplies like-for-like feature contributions for the model
        comparison's average-prediction section.  The reference artifact is
        the full-data/original model; the evaluated artifact may be that model
        itself or the edited model with selected edit offsets applied. The
        test-set method samples the exact held-out rows in the original JSON.
        The uniform method samples numeric curves on the shared 100-point
        metric grid over the two artifacts' overlapping domain.
        """
        row_based_method = method in {"test_set", "main_dataset"}
        if method == "test_set":
            sample_rows = (
                (reference_artifact.get("held_out_test_data") or {}).get("feature_rows")
                or []
            )
        if row_based_method and not sample_rows:
            return None
        missing_indicator_features = cls._compare_missing_indicator_features(
            reference_artifact, evaluated_artifact
        )
        public_features = set(reference_artifact.get("selected_feature_columns") or [])
        reference_shapes = [
            shape_function
            for shape_function in (reference_artifact.get("shape_functions") or [])
            if shape_function.get("feature_name") in public_features
            and shape_function.get("feature_name") not in missing_indicator_features
        ]
        if row_based_method:
            feature_names = dict.fromkeys(
                (reference_artifact.get("feature_names") or [])
                + (evaluated_artifact.get("feature_names") or [])
            )
            reference_shapes = [
                {"feature_name": feature_name}
                for feature_name in feature_names
                if feature_name not in missing_indicator_features
            ]
        evaluated_shape_map = {
            str(shape_function.get("feature_name", "")).strip(): shape_function
            for shape_function in (evaluated_artifact.get("shape_functions") or [])
        }
        numeric_grid_shape_map = {
            str(shape_function.get("feature_name", "")).strip(): shape_function
            for shape_function in (
                (numeric_grid_artifact or evaluated_artifact).get("shape_functions") or []
            )
        }
        evaluated_categorical_features = {
            str(feature_name)
            for feature_name in (evaluated_artifact.get("cat_features") or [])
        }
        contributions: Dict[str, float] = {}

        for reference_shape in reference_shapes:
            feature_name = str(reference_shape.get("feature_name", "")).strip()
            evaluated_shape = evaluated_shape_map.get(feature_name)
            if not feature_name:
                return None
            if not evaluated_shape:
                if row_based_method:
                    contributions[feature_name] = 0.0
                    continue
                return None

            if method == "uniform":
                reference_x_values = list(reference_shape.get("x_values") or [])
                reference_y_values = list(reference_shape.get("y_values") or [])
                if (
                    len(reference_x_values) != len(reference_y_values)
                    or not reference_x_values
                ):
                    return None
                if reference_shape.get("feature_type") == "numeric":
                    grid_shape = numeric_grid_shape_map.get(feature_name)
                    grid_x_values = list((grid_shape or {}).get("x_values") or [])
                    if not grid_x_values:
                        return None
                    reference_numeric_x = np.asarray(reference_x_values, dtype=float)
                    grid_numeric_x = np.asarray(grid_x_values, dtype=float)
                    overlap_min = max(
                        float(np.min(reference_numeric_x)),
                        float(np.min(grid_numeric_x)),
                    )
                    overlap_max = min(
                        float(np.max(reference_numeric_x)),
                        float(np.max(grid_numeric_x)),
                    )
                    if overlap_min > overlap_max:
                        return None
                    weighted_values = [
                        (value, 1)
                        for value in np.linspace(
                            overlap_min,
                            overlap_max,
                            MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS,
                        )
                    ]
                else:
                    weighted_values = [(value, 1) for value in reference_x_values]
            elif method == "data_weighted":
                distribution = reference_shape.get("distribution") or {}
                if distribution.get("chart_type") == "categorical":
                    weighted_values = [
                        (item.get("x_value"), int(item.get("count", 0) or 0))
                        for item in (distribution.get("counts") or [])
                        if int(item.get("count", 0) or 0) > 0
                    ]
                elif distribution.get("chart_type") == "numeric":
                    weighted_values = [
                        (item.get("center"), int(item.get("count", 0) or 0))
                        for item in (distribution.get("bins") or [])
                        if int(item.get("count", 0) or 0) > 0
                    ]
                else:
                    return None
            elif row_based_method:
                weighted_values = [
                    (row.get(feature_name), 1) for row in sample_rows
                ]
            else:
                raise ValueError(f"Unknown average contribution method '{method}'")

            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight <= 0:
                return None

            contribution = 0.0
            for value, weight in weighted_values:
                if evaluated_shape.get("feature_type") == "categorical":
                    base_value = cls._predict_categorical_shape_value(
                        evaluated_shape,
                        value,
                    )
                else:
                    base_value = cls._predict_numeric_shape_value(
                        evaluated_shape,
                        value,
                    )
                offset = cls._get_compare_offset_for_feature_value(
                    shape_function=evaluated_shape,
                    original_is_categorical=(
                        feature_name in evaluated_categorical_features
                    ),
                    value=value,
                    offsets=evaluated_offsets.get(feature_name, {}),
                )
                contribution += weight * (
                    base_value
                    + float(offset)
                    + float((feature_shifts or {}).get(feature_name, 0.0))
                )
            contributions[feature_name] = contribution / total_weight

        return contributions

    @staticmethod
    def _predict_numeric_shape_value(
        shape_function: Dict[str, Any],
        value: Any,
    ) -> float:
        try:
            x_values = np.asarray(shape_function.get("x_values") or [], dtype=float)
            y_values = np.asarray(shape_function.get("y_values") or [], dtype=float)
            return float(np.interp(float(value), x_values, y_values))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _predict_categorical_shape_value(
        shape_function: Dict[str, Any],
        value: Any,
    ) -> float:
        x_values = [str(item) for item in (shape_function.get("x_values") or [])]
        y_values = shape_function.get("y_values") or []
        try:
            return float(y_values[x_values.index(str(value))])
        except (ValueError, IndexError, TypeError):
            return 0.0

    @staticmethod
    def _scale_compare_offsets(
        offsets: Dict[str, Dict[Any, float]],
        scale_factor: float,
        *,
        feature_names: Optional[set[str]] = None,
    ) -> Dict[str, Dict[Any, float]]:
        """Scale selected edit offsets by one preview-only factor."""
        return {
            feature_name: {
                value: (
                    float(offset) * scale_factor
                    if feature_names is None or feature_name in feature_names
                    else float(offset)
                )
                for value, offset in point_offsets.items()
            }
            for feature_name, point_offsets in offsets.items()
        }

    @staticmethod
    def _scale_compare_feature_shifts(
        feature_shifts: Dict[str, float],
        scale_factor: float,
        *,
        feature_names: Optional[set[str]] = None,
    ) -> Dict[str, float]:
        """Scale preview-only whole-feature shifts for the selected features."""
        return {
            feature_name: (
                float(shift) * scale_factor
                if feature_names is None or feature_name in feature_names
                else float(shift)
            )
            for feature_name, shift in feature_shifts.items()
        }

    @classmethod
    def _build_overall_shape_function_mae(
        cls,
        per_feature_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarize full-data discrepancy across the compared shape functions.

        ``pooled`` weights a feature by the number of full-data reference-grid
        points used for its MAE. ``mean_per_feature`` gives every compared
        shape function the same weight, regardless of its displayed grid size.
        """
        valid_metrics = [
            metric
            for metric in per_feature_metrics
            if int(metric.get("point_count", 0) or 0) > 0
            and np.isfinite(metric.get("baseline", np.nan))
            and np.isfinite(metric.get("edited", np.nan))
        ]
        feature_count = len(valid_metrics)
        point_count = sum(int(metric["point_count"]) for metric in valid_metrics)

        def summary(baseline: float, edited: float) -> Dict[str, Any]:
            return {
                "baseline": baseline,
                "edited": edited,
                "status": cls._compare_mae_status(edited, baseline),
            }

        if not valid_metrics or point_count <= 0:
            return {
                "feature_count": 0,
                "point_count": 0,
                "pooled": summary(0.0, 0.0),
                "mean_per_feature": summary(0.0, 0.0),
            }

        pooled_baseline = sum(
            float(metric["baseline"]) * int(metric["point_count"])
            for metric in valid_metrics
        ) / point_count
        pooled_edited = sum(
            float(metric["edited"]) * int(metric["point_count"])
            for metric in valid_metrics
        ) / point_count
        mean_baseline = sum(
            float(metric["baseline"]) for metric in valid_metrics
        ) / feature_count
        mean_edited = sum(
            float(metric["edited"]) for metric in valid_metrics
        ) / feature_count
        return {
            "feature_count": feature_count,
            "point_count": point_count,
            "pooled": summary(pooled_baseline, pooled_edited),
            "mean_per_feature": summary(mean_baseline, mean_edited),
        }

    @classmethod
    def _build_expert_complete_model_mae(
        cls,
        per_expert_feature_metrics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Pool each expert's model MAE over every selected shape function."""
        summaries: Dict[str, Dict[str, Any]] = {}
        for metric in per_expert_feature_metrics:
            point_count = int(metric.get("point_count", 0) or 0)
            mae = float(metric.get("mae", np.nan))
            baseline_mae = float(metric.get("baseline_mae", np.nan))
            user_name = str(metric.get("user_name", "") or "").strip()
            if (
                not user_name
                or point_count <= 0
                or not np.isfinite(mae)
                or not np.isfinite(baseline_mae)
            ):
                continue
            summary = summaries.setdefault(
                user_name,
                {
                    "user_name": user_name,
                    "mae_sum": 0.0,
                    "baseline_mae_sum": 0.0,
                    "feature_count": 0,
                    "point_count": 0,
                    "submission_ids": set(),
                },
            )
            summary["mae_sum"] += mae * point_count
            summary["baseline_mae_sum"] += baseline_mae * point_count
            summary["feature_count"] += 1
            summary["point_count"] += point_count
            summary["submission_ids"].update(metric.get("submission_ids", set()))

        return [
            {
                "user_name": summary["user_name"],
                "mae": summary["mae_sum"] / summary["point_count"],
                "baseline_mae": summary["baseline_mae_sum"]
                / summary["point_count"],
                "status": cls._compare_mae_status(
                    summary["mae_sum"] / summary["point_count"],
                    summary["baseline_mae_sum"] / summary["point_count"],
                ),
                "feature_count": summary["feature_count"],
                "point_count": summary["point_count"],
                "submission_count": len(summary["submission_ids"]),
            }
            for summary in sorted(
                summaries.values(),
                key=lambda summary: summary["user_name"].casefold(),
            )
            if summary["point_count"] > 0
        ]

    def _build_all_feature_mae_change(
        self,
        *,
        left_artifact: Dict[str, Any],
        right_artifact: Dict[str, Any],
        shared_features: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compare all exported edits for each feature in both confidence modes."""
        left_shapes = {
            str(shape.get("feature_name")): shape
            for shape in left_artifact.get("shape_functions") or []
        }
        right_shapes = {
            str(shape.get("feature_name")): shape
            for shape in right_artifact.get("shape_functions") or []
        }
        right_categorical = set(right_artifact.get("cat_features") or [])
        results: Dict[str, List[Dict[str, Any]]] = {}

        for mode, use_confidence in (
            ("without_confidence", False),
            ("with_confidence", True),
        ):
            offsets = self._collect_compare_offsets(
                right_artifact, None, use_confidence=use_confidence
            )
            feature_metrics = []
            for feature_name in shared_features:
                left_shape = left_shapes.get(feature_name)
                right_shape = right_shapes.get(feature_name)
                if not left_shape or not right_shape:
                    continue
                left_x = list(left_shape.get("x_values") or [])
                right_x = list(right_shape.get("x_values") or [])
                left_y = [float(value) for value in left_shape.get("y_values") or []]
                right_y = [float(value) for value in right_shape.get("y_values") or []]
                if (
                    not left_x
                    or not right_x
                    or len(left_x) != len(left_y)
                    or len(right_x) != len(right_y)
                ):
                    continue

                edited_y = [
                    base_y
                    + self._get_compare_offset_for_feature_value(
                        shape_function=right_shape,
                        original_is_categorical=feature_name in right_categorical,
                        value=x_value,
                        offsets=offsets.get(feature_name, {}),
                    )
                    for x_value, base_y in zip(right_x, right_y)
                ]
                if str(left_shape.get("feature_type", "numeric")) == "numeric":
                    left_numeric_x = np.asarray(left_x, dtype=float)
                    right_numeric_x = np.asarray(right_x, dtype=float)
                    overlap_min = max(
                        float(np.min(left_numeric_x)), float(np.min(right_numeric_x))
                    )
                    overlap_max = min(
                        float(np.max(left_numeric_x)), float(np.max(right_numeric_x))
                    )
                    if overlap_min > overlap_max:
                        raise ValueError(
                            f"Cannot compute all-feature MAE change without overlapping numeric domain for '{feature_name}'"
                        )
                    sample_x = np.linspace(
                        overlap_min,
                        overlap_max,
                        MODEL_COMPARE_NUMERIC_FEATURE_METRIC_POINTS,
                    ).tolist()
                    reference_y = self._interpolate_compare_values(
                        left_x, left_y, sample_x
                    )
                    baseline_y = self._interpolate_compare_values(
                        right_x, right_y, sample_x
                    )
                    sampled_edited_y = self._interpolate_compare_values(
                        right_x, edited_y, sample_x
                    )
                else:
                    reference_y = left_y
                    baseline_y = right_y
                    sampled_edited_y = edited_y
                baseline_mae = self._compute_compare_mae(reference_y, baseline_y)
                edited_mae = self._compute_compare_mae(reference_y, sampled_edited_y)
                feature_metrics.append(
                    {
                        "feature_name": feature_name,
                        "mae": edited_mae,
                        "baseline_mae": baseline_mae,
                        "status": self._compare_mae_status(edited_mae, baseline_mae),
                        "point_count": len(reference_y),
                    }
                )
            results[mode] = feature_metrics
        return results
