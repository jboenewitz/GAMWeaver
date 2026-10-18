"""Store and apply interactive shape function offsets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd

from .noop_edits import encode_numeric_storage_key


class InteractiveEditsMixin:
    @staticmethod
    def _interpolate_curve_value(
        x_values: List[Any],
        y_values: List[Any],
        target_x: float,
    ) -> float:
        """Linearly interpolate y on a curve at target_x."""
        if not x_values or not y_values or len(x_values) != len(y_values):
            return 0.0

        try:
            pairs = sorted(
                [(float(x), float(y)) for x, y in zip(x_values, y_values)],
                key=lambda item: item[0],
            )
        except (TypeError, ValueError):
            return 0.0

        if not pairs:
            return 0.0
        if len(pairs) == 1:
            return pairs[0][1]

        if target_x <= pairs[0][0]:
            return pairs[0][1]
        if target_x >= pairs[-1][0]:
            return pairs[-1][1]

        for idx in range(len(pairs) - 1):
            left_x, left_y = pairs[idx]
            right_x, right_y = pairs[idx + 1]
            if left_x <= target_x <= right_x:
                span = right_x - left_x
                if span == 0:
                    return left_y
                ratio = (target_x - left_x) / span
                return left_y + ratio * (right_y - left_y)

        return pairs[-1][1]

    @staticmethod
    def _encode_numeric_storage_key(x_value: float) -> str:
        """Encode numeric x values with an explicit prefix for persistence."""
        return encode_numeric_storage_key(x_value)

    @staticmethod
    def _decode_numeric_storage_key(raw_key: Any, original_x: List[Any]) -> Optional[float]:
        """
        Decode numeric storage key.

        Supported formats:
        - Legacy index (int / numeric string)
        - Explicit coordinate ("x:<float>")
        - Fallback direct float value
        """
        if isinstance(raw_key, (int, np.integer)):
            idx = int(raw_key)
            if 0 <= idx < len(original_x):
                return float(original_x[idx])
            return None

        if isinstance(raw_key, (float, np.floating)):
            return float(raw_key)

        raw_str = str(raw_key).strip()
        if not raw_str:
            return None

        if raw_str.startswith("x:"):
            try:
                return float(raw_str[2:])
            except ValueError:
                return None

        try:
            idx = int(raw_str)
            if 0 <= idx < len(original_x):
                return float(original_x[idx])
        except ValueError:
            pass

        try:
            return float(raw_str)
        except ValueError:
            return None

    def _decode_numeric_offsets_for_feature(
        self,
        feature_name: str,
        offsets: Dict[Any, float],
    ) -> List[Tuple[float, float]]:
        """Convert stored/legacy numeric offsets into sorted (x, offset) points."""
        original = self.original_shape_functions.get(feature_name, {})
        original_x = original.get("x_values", [])

        point_map: Dict[float, float] = {}
        for raw_key, raw_offset in (offsets or {}).items():
            x_val = self._decode_numeric_storage_key(raw_key, original_x)
            if x_val is None:
                continue
            try:
                point_map[float(x_val)] = float(raw_offset)
            except (TypeError, ValueError):
                continue

        return sorted(point_map.items(), key=lambda item: item[0])

    def load_shape_function_offsets_from_storage(
        self,
        storage_edits: List[Dict[str, Any]],
    ) -> None:
        """Load persisted edits into runtime offsets (supports legacy and new formats)."""
        self.shape_function_offsets = {}

        for stored_sf in storage_edits:
            feature_name = stored_sf.get("feature_name")
            feature_type = stored_sf.get("feature_type")
            stored_points = stored_sf.get("edited_points", [])

            if not feature_name or feature_name not in self.original_shape_functions:
                continue

            self.shape_function_offsets[feature_name] = {}
            original_x = self.original_shape_functions[feature_name]["x_values"]

            for stored_point in stored_points:
                x_val = stored_point.get("x_value")
                offset = stored_point.get("y_value", 0.0)
                try:
                    offset_float = float(offset)
                except (TypeError, ValueError):
                    continue

                if feature_type == "categorical":
                    self.shape_function_offsets[feature_name][str(x_val)] = offset_float
                else:
                    decoded_x = self._decode_numeric_storage_key(x_val, original_x)
                    if decoded_x is None:
                        continue
                    self.shape_function_offsets[feature_name][float(decoded_x)] = offset_float

    def update_shape_functions(self, edited_shape_functions: List[Dict[str, Any]]) -> None:
        """Update shape function offsets based on user edits."""
        self.shape_function_offsets = {}

        for edited_sf in edited_shape_functions:
            feature_name = edited_sf["feature_name"]
            feature_type = edited_sf["feature_type"]
            edited_points = edited_sf["edited_points"]

            if feature_name not in self.original_shape_functions:
                continue

            original = self.original_shape_functions[feature_name]
            original_x = original["x_values"]
            original_y = original["y_values"]
            self.shape_function_offsets[feature_name] = {}

            for edited_point in edited_points:
                x_val = edited_point["x_value"]
                new_y = edited_point["y_value"]

                if feature_type == "categorical":
                    x_str = str(x_val)
                    if x_str in original_x:
                        idx = original_x.index(x_str)
                        original_y_val = original_y[idx]
                        offset = new_y - original_y_val
                        self.shape_function_offsets[feature_name][x_str] = offset
                else:
                    x_float = float(x_val)
                    original_y_val = self._interpolate_curve_value(original_x, original_y, x_float)
                    offset = new_y - original_y_val
                    self.shape_function_offsets[feature_name][x_float] = offset

    def convert_edits_for_storage(
        self,
        edited_shape_functions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert edited shape functions to storage format.

        Numeric features: x_value -> encoded coordinate key, y_value -> offset.
        Categorical features: x_value string, y_value -> offset.
        """
        result = []
        for edited_sf in edited_shape_functions:
            feature_name = edited_sf["feature_name"]
            feature_type = edited_sf["feature_type"]
            edited_points = edited_sf["edited_points"]

            if feature_name not in self.original_shape_functions:
                continue

            original = self.original_shape_functions[feature_name]
            original_x = original["x_values"]
            original_y = original["y_values"]

            converted_points = []
            for edited_point in edited_points:
                x_val = edited_point["x_value"]
                new_y = edited_point["y_value"]
                weight = edited_point.get("weight", 0.5)
                message = edited_point.get("message", "")

                if feature_type == "categorical":
                    x_str = str(x_val)
                    if x_str in original_x:
                        idx = original_x.index(x_str)
                        offset = new_y - original_y[idx]
                        converted_points.append(
                            {
                                "x_value": x_str,
                                "y_value": offset,
                                "weight": weight,
                                "message": message,
                            }
                        )
                else:
                    x_float = float(x_val)
                    offset = new_y - self._interpolate_curve_value(original_x, original_y, x_float)
                    converted_points.append(
                        {
                            "x_value": self._encode_numeric_storage_key(x_float),
                            "y_value": offset,
                            "weight": weight,
                            "message": message,
                        }
                    )

            if converted_points:
                result.append(
                    {
                        "feature_name": feature_name,
                        "feature_type": feature_type,
                        "edited_points": converted_points,
                    }
                )
        return result

    def convert_storage_to_display_format(
        self,
        storage_edits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert stored edits back to display format for frontend rendering."""
        result = []
        for stored_sf in storage_edits:
            feature_name = stored_sf["feature_name"]
            feature_type = stored_sf["feature_type"]
            stored_points = stored_sf.get("edited_points", [])

            if feature_name not in self.original_shape_functions:
                continue

            original = self.original_shape_functions[feature_name]
            original_x = original["x_values"]
            original_y = original["y_values"]

            display_points = []
            for stored_point in stored_points:
                x_val = stored_point["x_value"]
                offset = stored_point["y_value"]

                if feature_type == "categorical":
                    x_str = str(x_val)
                    if x_str in original_x:
                        idx = original_x.index(x_str)
                        display_points.append(
                            {
                                "x_value": x_str,
                                "y_value": original_y[idx] + offset,
                            }
                        )
                else:
                    decoded_x = self._decode_numeric_storage_key(x_val, original_x)
                    if decoded_x is None:
                        continue
                    base_y = self._interpolate_curve_value(original_x, original_y, decoded_x)
                    display_points.append(
                        {
                            "x_value": float(decoded_x),
                            "y_value": base_y + offset,
                        }
                    )

            if display_points:
                result.append(
                    {
                        "feature_name": feature_name,
                        "feature_type": feature_type,
                        "edited_points": display_points,
                    }
                )
        return result

    def _get_offset_for_value(self, feature_name: str, value: Any) -> float:
        """Get interpolated offset for a feature value."""
        if feature_name not in self.shape_function_offsets:
            return 0.0

        offsets = self.shape_function_offsets[feature_name]
        return self.get_offset_for_feature_value(feature_name, value, offsets)

    def get_offset_for_feature_value(
        self,
        feature_name: str,
        value: Any,
        offsets: Optional[Dict[Any, float]] = None,
    ) -> float:
        """Get interpolated offset for a feature value from a given offsets map."""
        offsets = offsets if offsets is not None else self.shape_function_offsets.get(feature_name, {})
        if not offsets:
            return 0.0

        is_categorical_chart = self._is_feature_categorical_for_chart(feature_name)
        if is_categorical_chart:
            candidate_keys = [
                self._stringify_chart_value(value),
                str(value),
            ]
            for raw_key in candidate_keys:
                if raw_key in offsets:
                    try:
                        return float(offsets.get(raw_key, 0.0))
                    except (TypeError, ValueError):
                        return 0.0

            # Backward compatibility: if old numeric-style keys exist, use exact numeric match.
            points = self._decode_numeric_offsets_for_feature(feature_name, offsets)
            if points:
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    return 0.0
                for x_val, x_offset in points:
                    if abs(x_val - value_float) < 1e-9:
                        return float(x_offset)
            return 0.0

        # Numeric chart mode on originally categorical features:
        # allow exact string key lookup (from previously stored categorical edits)
        # before attempting numeric interpolation.
        if feature_name in self.cat_features:
            candidate_keys = [
                self._stringify_chart_value(value),
                str(value),
            ]
            for raw_key in candidate_keys:
                if raw_key in offsets:
                    try:
                        return float(offsets.get(raw_key, 0.0))
                    except (TypeError, ValueError):
                        return 0.0

        points = self._decode_numeric_offsets_for_feature(feature_name, offsets)
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
                t_val = (value_float - left_x) / span
                return float(left_offset + t_val * (right_offset - left_offset))

        return 0.0

    def predict_interactive(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with interactive shape function modifications."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        base_predictions = self.model.predict(X)
        offsets = np.zeros(len(X))

        for feature_name in self.shape_function_offsets:
            if feature_name in X.columns:
                for idx, (_, row) in enumerate(X.iterrows()):
                    offsets[idx] += self._get_offset_for_row(feature_name, row)

        return base_predictions + offsets

    def get_predictions_comparison(self) -> Dict[str, Any]:
        """Get comparison between original and interactive predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        self._require_analytics_data()

        y_pred_original = self.model.predict(self.X_test)
        y_pred_interactive = self.predict_interactive(self.X_test)
        y_actual = self.y_test.values.flatten()

        original_rmse = root_mean_squared_error(y_actual, y_pred_original)
        original_mae = mean_absolute_error(y_actual, y_pred_original)
        interactive_rmse = root_mean_squared_error(y_actual, y_pred_interactive)
        interactive_mae = mean_absolute_error(y_actual, y_pred_interactive)

        return {
            "original_predictions": [float(value) for value in y_pred_original],
            "interactive_predictions": [float(value) for value in y_pred_interactive],
            "actual_values": [float(value) for value in y_actual],
            "metrics": {
                "original_rmse": float(original_rmse),
                "original_mae": float(original_mae),
                "interactive_rmse": float(interactive_rmse),
                "interactive_mae": float(interactive_mae),
            },
        }
