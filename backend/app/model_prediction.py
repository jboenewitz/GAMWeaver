"""Predictions, shape functions, model evaluation, and data summaries."""

from __future__ import annotations

from igann import IGANN
from typing import Any, Dict, List, Optional

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd

from .data_processing import MISSING_CATEGORY_VALUE
from .ml_shared import SHAPE_DISTRIBUTION_DEFAULT_BIN_COUNT


class ModelPredictionMixin:
    def train_model(self, n_estimators: int = 100) -> Dict[str, float]:
        """Train the IGANN model."""
        if self.X_train is None:
            self.load_data()

        self.model = IGANN(
            task="regression",
            n_estimators=n_estimators,
            verbose=0,
            scale_y=True,
        )
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        self.model_source = "trained"
        self.imported_artifact_version = None
        self.analytics_available = self._has_analytics_data()
        self.primary_n_estimators = int(n_estimators)
        self.original_shape_functions = {}
        self.shape_function_offsets = {}
        self._clear_comparison_state()

        metrics = self.evaluate_model()
        return metrics

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        self._require_analytics_data()

        y_pred = self.model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "model_type": "IGANN",
        }

    def _public_feature_names(self) -> List[str]:
        return (
            list(self.selected_feature_columns)
            if self.selected_feature_columns
            else list(self.feature_schema_map.keys())
        )

    def _missing_indicator_feature_name(self, feature_name: str) -> Optional[str]:
        return self.missing_indicator_map.get(feature_name)

    def _is_missing_category_value(self, value: Any) -> bool:
        return str(value) == MISSING_CATEGORY_VALUE

    def _is_missing_numeric_feature_in_row(
        self,
        feature_name: str,
        row: Any,
    ) -> bool:
        indicator_name = self._missing_indicator_feature_name(feature_name)
        if not indicator_name:
            return False
        try:
            indicator_value = row[indicator_name]
        except Exception:
            try:
                indicator_value = row.get(indicator_name)
            except Exception:
                indicator_value = None
        return self._is_missing_category_value(indicator_value)

    def _get_public_training_series(self, feature_name: str) -> pd.Series:
        if (
            self.df is None
            or self.X_train is None
            or feature_name not in self.selected_feature_columns
            or feature_name not in self.df.columns
        ):
            return pd.Series(dtype=object)
        return self.df.loc[self.X_train.index, feature_name]

    def _get_public_missing_count(self, feature_name: str) -> int:
        raw_series = self._get_public_training_series(feature_name)
        if raw_series.empty:
            return 0
        if feature_name in self.num_features:
            return int(pd.to_numeric(raw_series, errors="coerce").isna().sum())
        return int(raw_series.isna().sum())

    def _normalize_public_series_for_categorical_chart(
        self,
        feature_name: str,
    ) -> pd.Series:
        raw_series = self._get_public_training_series(feature_name)
        if raw_series.empty:
            return pd.Series(dtype=object)
        if feature_name in self.num_features:
            numeric = pd.to_numeric(raw_series, errors="coerce")
            return numeric.map(
                lambda value: (
                    MISSING_CATEGORY_VALUE
                    if pd.isna(value)
                    else self._stringify_chart_value(value)
                )
            )
        return (
            raw_series.astype(object)
            .where(raw_series.notna(), MISSING_CATEGORY_VALUE)
            .astype(str)
        )

    def _predict_shape_function_sample(self, sample_data: Dict[str, Any]) -> float:
        sample = pd.DataFrame([sample_data])[self.feature_names]

        for cat_feat in self.cat_features:
            sample[cat_feat] = sample[cat_feat].astype(str)
        for num_feat in self.num_features:
            sample[num_feat] = sample[num_feat].astype(float)

        pred = self.model.predict(sample)
        return float(pred[0]) if hasattr(pred, "__iter__") else float(pred)

    def _build_numeric_chart_missing_bucket(
        self,
        feature_name: str,
        baseline: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        missing_count = self._get_public_missing_count(feature_name)
        if missing_count <= 0:
            return None

        sample_data = baseline.copy()
        if feature_name in self.num_features:
            indicator_name = self._missing_indicator_feature_name(feature_name)
            if not indicator_name:
                return None
            sample_data[feature_name] = float(
                self.numeric_missing_placeholder_values.get(
                    feature_name,
                    self.feature_schema_map.get(feature_name, {}).get(
                        "default_value",
                        0.0,
                    ),
                )
            )
            sample_data[indicator_name] = MISSING_CATEGORY_VALUE
        elif feature_name in self.cat_features:
            sample_data[feature_name] = MISSING_CATEGORY_VALUE
        else:
            return None

        prediction = self._predict_shape_function_sample(sample_data)
        baseline_prediction = self._predict_shape_function_sample(baseline)
        return {
            "label": MISSING_CATEGORY_VALUE,
            "count": missing_count,
            "y_value": prediction - baseline_prediction,
        }

    def _build_model_row_from_public_features(
        self,
        normalized_public_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for feature_name in self._public_feature_names():
            schema = self.feature_schema_map.get(feature_name, {})
            feature_type = schema.get("feature_type")
            value = normalized_public_features.get(feature_name)

            if feature_type == "numeric":
                indicator_name = self._missing_indicator_feature_name(feature_name)
                if value is None:
                    row[feature_name] = float(
                        self.numeric_missing_placeholder_values.get(feature_name, 0.0)
                    )
                    if indicator_name:
                        row[indicator_name] = MISSING_CATEGORY_VALUE
                else:
                    row[feature_name] = float(value)
                    if indicator_name:
                        row[indicator_name] = "Observed"
            else:
                row[feature_name] = (
                    MISSING_CATEGORY_VALUE if value is None else str(value)
                )

        return row

    def _validate_and_normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction payload against loaded feature schema."""
        if not self.feature_schema_map:
            raise ValueError("Feature schema is not available yet")

        provided_keys = set(features.keys())
        expected_keys = set(self._public_feature_names())
        missing = sorted(expected_keys - provided_keys)
        extra = sorted(provided_keys - expected_keys)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(f"Prediction features do not match dataset schema ({', '.join(details)})")

        normalized: Dict[str, Any] = {}
        for feature_name in self._public_feature_names():
            value = features[feature_name]
            schema = self.feature_schema_map.get(feature_name, {})
            feature_type = schema.get("feature_type")

            if feature_type == "numeric":
                if value is None or (isinstance(value, str) and not value.strip()):
                    normalized[feature_name] = None
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature '{feature_name}' must be numeric") from exc
                if not np.isfinite(numeric_value):
                    normalized[feature_name] = None
                    continue
                normalized[feature_name] = numeric_value
            else:
                if value is None or (isinstance(value, str) and not value.strip()):
                    normalized[feature_name] = None
                    continue
                value_str = str(value)
                valid_options = schema.get("categorical_options") or []
                if valid_options and value_str not in valid_options:
                    raise ValueError(
                        f"Feature '{feature_name}' value '{value_str}' is not in allowed options"
                    )
                normalized[feature_name] = value_str

        return normalized

    def _build_input_df(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Build a single-row DataFrame from dynamic prediction features."""
        normalized = self._validate_and_normalize_features(features)
        input_df = pd.DataFrame([self._build_model_row_from_public_features(normalized)])
        for cat_feature in self.cat_features:
            if cat_feature in input_df.columns:
                input_df[cat_feature] = input_df[cat_feature].astype(str)
        for num_feature in self.num_features:
            if num_feature in input_df.columns:
                input_df[num_feature] = pd.to_numeric(input_df[num_feature], errors="coerce")
        return input_df

    def _get_offset_for_row(self, feature_name: str, row: Any) -> float:
        """Get a feature offset for one row, skipping numeric offsets on missing values."""
        if self._is_missing_numeric_feature_in_row(feature_name, row):
            return 0.0
        try:
            value = row[feature_name]
        except Exception:
            try:
                value = row.get(feature_name)
            except Exception:
                return 0.0
        return self._get_offset_for_value(feature_name, value)

    def predict(self, features: Dict[str, Any]) -> float:
        """Make a single prediction, applying current shape function offsets."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        input_df = self._build_input_df(features)
        base_prediction = float(self.model.predict(input_df)[0])

        total_offset = 0.0
        for feature_name in self.shape_function_offsets:
            if feature_name in input_df.columns:
                total_offset += self._get_offset_for_row(feature_name, input_df.iloc[0])

        return base_prediction + total_offset

    def predict_with_offsets(
        self,
        features: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
    ) -> float:
        """Make a single prediction using provided shape-function offsets."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        input_df = self._build_input_df(features)
        base_prediction = float(self.model.predict(input_df)[0])

        original_offsets = self.shape_function_offsets
        self.shape_function_offsets = offsets

        total_offset = 0.0
        for feature_name in offsets:
            if feature_name in input_df.columns:
                total_offset += self._get_offset_for_row(feature_name, input_df.iloc[0])

        self.shape_function_offsets = original_offsets
        return base_prediction + total_offset

    def batch_predict(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Make batch predictions."""
        return [self.predict(features) for features in features_list]

    def get_shape_functions(self, include_distribution: bool = True) -> List[Dict[str, Any]]:
        """Get shape function data for all features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        if self.model_source == "imported" and self.original_shape_functions:
            return list(self.original_shape_functions.values())

        shape_functions = []
        for idx, feature_name in enumerate(self._public_feature_names()):
            try:
                shape_data = self._extract_shape_function(
                    feature_name,
                    idx,
                    include_distribution=include_distribution,
                )
                shape_functions.append(shape_data)
                self.original_shape_functions[feature_name] = shape_data
            except Exception as exc:
                print(f"Error extracting shape function for {feature_name}: {exc}")
                continue

        self.shape_function_offsets = {}
        return shape_functions

    def get_comparison_shape_functions(self) -> List[Dict[str, Any]]:
        """Get shape function data for the comparison model, aligned to primary features."""
        if not self.comparison_is_trained or self.comparison_model is None:
            return []

        snapshot = self._snapshot_runtime_state()
        try:
            self.model = self.comparison_model
            self.preprocessor = self.comparison_preprocessor
            self.X_train = self.comparison_X_train
            self.X_test = self.comparison_X_test
            self.y_train = self.comparison_y_train
            self.y_test = self.comparison_y_test
            self.df = self.comparison_df
            self.is_trained = self.comparison_is_trained
            self.feature_names = list(self.comparison_feature_names)
            self.selected_feature_columns = list(
                self.comparison_selected_feature_columns
            )
            self.cat_features = list(self.comparison_cat_features)
            self.num_features = list(self.comparison_num_features)
            self.target_column = self.comparison_target_column
            self.feature_schema = list(self.comparison_feature_schema)
            self.feature_schema_map = dict(self.comparison_feature_schema_map)
            self.missing_indicator_map = dict(self.comparison_missing_indicator_map)
            self.numeric_missing_placeholder_values = dict(
                self.comparison_numeric_missing_placeholder_values
            )
            self.original_shape_functions = dict(
                self.comparison_original_shape_functions
            )
            self.shape_function_offsets = {
                feature: dict(offsets)
                for feature, offsets in self.comparison_shape_function_offsets.items()
            }
            self.model_source = "trained"
            self.imported_artifact_version = None
            self.analytics_available = False

            shape_functions = self.get_shape_functions(include_distribution=False)
            self.comparison_original_shape_functions = {
                shape_function["feature_name"]: shape_function
                for shape_function in shape_functions
            }
            self.comparison_shape_function_offsets = {}
            return shape_functions
        finally:
            self._restore_runtime_state(snapshot)

    def _build_shape_baseline(self) -> Dict[str, Any]:
        baseline: Dict[str, Any] = {}
        for num_feat in self.num_features:
            numeric_series = pd.to_numeric(
                self._get_public_training_series(num_feat),
                errors="coerce",
            ).dropna()
            if numeric_series.empty:
                baseline[num_feat] = float(
                    self.feature_schema_map.get(num_feat, {}).get("default_value", 0.0)
                )
            else:
                baseline[num_feat] = float(numeric_series.mean())
        for cat_feat in self.cat_features:
            mode = self.X_train[cat_feat].astype(str).mode()
            baseline[cat_feat] = str(mode.iloc[0]) if not mode.empty else "Unknown"
        return baseline

    def _build_numeric_distribution_bins(
        self,
        values: pd.Series,
        x_values: List[Any],
        missing_count: int = 0,
    ) -> Optional[Dict[str, Any]]:
        numeric_values = pd.to_numeric(values, errors="coerce").dropna()
        if numeric_values.empty:
            return None

        displayed_domain = [
            float(value)
            for value in x_values
            if value is not None and np.isfinite(pd.to_numeric(value, errors="coerce"))
        ]
        if displayed_domain:
            domain_min = float(min(displayed_domain))
            domain_max = float(max(displayed_domain))
        else:
            domain_min = float(numeric_values.min())
            domain_max = float(numeric_values.max())

        unique_count = int(numeric_values.nunique(dropna=True))
        if unique_count <= 1 or abs(domain_max - domain_min) < 1e-12:
            center = float(numeric_values.iloc[0])
            half_width = 0.5
            return {
                "chart_type": "numeric",
                "total_count": int(len(numeric_values) + missing_count),
                "bin_count": 1,
                "bins": [
                    {
                        "x0": float(center - half_width),
                        "x1": float(center + half_width),
                        "count": int(len(numeric_values)),
                        "center": center,
                    }
                ],
                "counts": [],
                "missing_count": int(missing_count),
                "missing_label": (
                    MISSING_CATEGORY_VALUE if missing_count > 0 else None
                ),
            }

        edges = np.linspace(
            domain_min,
            domain_max,
            SHAPE_DISTRIBUTION_DEFAULT_BIN_COUNT + 1,
        )
        counts, _ = np.histogram(numeric_values.to_numpy(dtype=float), bins=edges)
        bins = []
        for idx, count in enumerate(counts.tolist()):
            x0 = float(edges[idx])
            x1 = float(edges[idx + 1])
            bins.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "count": int(count),
                    "center": float((x0 + x1) / 2),
                }
            )

        return {
            "chart_type": "numeric",
            "total_count": int(len(numeric_values) + missing_count),
            "bin_count": len(bins),
            "bins": bins,
            "counts": [],
            "missing_count": int(missing_count),
            "missing_label": (
                MISSING_CATEGORY_VALUE if missing_count > 0 else None
            ),
        }

    def _build_numeric_distribution_for_categorical_feature(
        self,
        feature_name: str,
    ) -> Optional[Dict[str, Any]]:
        numeric_pairs = self._get_numeric_chart_values_for_categorical_feature(feature_name)
        if not numeric_pairs:
            return None

        normalized_series = self._normalize_public_series_for_categorical_chart(
            feature_name,
        )
        counts_map = normalized_series.value_counts().to_dict()
        missing_count = int(counts_map.get(MISSING_CATEGORY_VALUE, 0))
        centers = [float(numeric_x) for _, numeric_x in numeric_pairs]
        if len(centers) == 1:
            edges = [centers[0] - 0.5, centers[0] + 0.5]
        else:
            edges = [centers[0] - ((centers[1] - centers[0]) / 2)]
            for idx in range(len(centers) - 1):
                edges.append((centers[idx] + centers[idx + 1]) / 2)
            edges.append(centers[-1] + ((centers[-1] - centers[-2]) / 2))

        bins = []
        total_count = 0
        for idx, (raw_value, numeric_x) in enumerate(numeric_pairs):
            count = int(counts_map.get(str(raw_value), 0))
            total_count += count
            bins.append(
                {
                    "x0": float(edges[idx]),
                    "x1": float(edges[idx + 1]),
                    "count": count,
                    "center": float(numeric_x),
                }
            )

        return {
            "chart_type": "numeric",
            "total_count": total_count + missing_count,
            "bin_count": len(bins),
            "bins": bins,
            "counts": [],
            "missing_count": missing_count,
            "missing_label": (
                MISSING_CATEGORY_VALUE if missing_count > 0 else None
            ),
        }

    def _build_categorical_distribution(
        self,
        feature_name: str,
        x_values: List[Any],
        x_tick_labels: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_series = self._normalize_public_series_for_categorical_chart(
            feature_name,
        )

        counts_map = normalized_series.value_counts().to_dict()
        ordered_counts = []
        total_count = 0
        labels = (
            list(x_tick_labels)
            if isinstance(x_tick_labels, list) and len(x_tick_labels) == len(x_values)
            else None
        )

        for idx, raw_value in enumerate(x_values):
            normalized_value = self._stringify_chart_value(raw_value)
            count = int(counts_map.get(normalized_value, 0))
            total_count += count
            ordered_counts.append(
                {
                    "x_value": raw_value,
                    "label": labels[idx] if labels is not None else normalized_value,
                    "count": count,
                }
            )

        return {
            "chart_type": "categorical",
            "total_count": total_count,
            "bin_count": len(ordered_counts),
            "bins": [],
            "counts": ordered_counts,
            "missing_count": int(counts_map.get(MISSING_CATEGORY_VALUE, 0)),
            "missing_label": MISSING_CATEGORY_VALUE,
        }

    def _build_shape_function_distribution(
        self,
        feature_name: str,
        feature_type: str,
        x_values: List[Any],
        x_tick_labels: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if feature_name not in self._public_feature_names():
            return None

        if feature_type == "categorical":
            return self._build_categorical_distribution(
                feature_name,
                x_values,
                x_tick_labels=x_tick_labels,
            )

        if feature_name in self.cat_features:
            return self._build_numeric_distribution_for_categorical_feature(
                feature_name,
            )

        return self._build_numeric_distribution_bins(
            self._get_public_training_series(feature_name),
            x_values,
            missing_count=self._get_public_missing_count(feature_name),
        )

    def _extract_shape_function(
        self,
        feature_name: str,
        feature_idx: int,
        include_distribution: bool = True,
    ) -> Dict[str, Any]:
        """Extract shape function data for a single feature."""
        chart_setting = self.get_feature_chart_setting(feature_name)
        is_categorical = chart_setting.get("chart_feature_type") == "categorical"
        categorical_value_labels = (
            chart_setting.get("categorical_value_labels", {}) or {}
        )
        numeric_tick_labels = chart_setting.get("numeric_tick_labels", {}) or {}
        baseline = self._build_shape_baseline()
        if self.model is None or not hasattr(self.model, "get_shape_functions_as_dict"):
            raise ValueError("Native model shape functions are unavailable")
        native_shape = self.model.get_shape_functions_as_dict().get(feature_name)
        if not native_shape:
            raise ValueError(f"No native shape function is available for '{feature_name}'")

        raw_native_x_values = native_shape.get("x", [])
        raw_native_y_values = native_shape.get("y", [])
        native_x_values = (
            list(raw_native_x_values) if raw_native_x_values is not None else []
        )
        native_y_values = (
            list(raw_native_y_values) if raw_native_y_values is not None else []
        )
        if len(native_x_values) != len(native_y_values) or not native_x_values:
            raise ValueError(f"Native shape function for '{feature_name}' is invalid")
        native_is_categorical = (
            str(native_shape.get("datatype", "")).strip() == "categorical"
        )

        def native_effect_at(value: Any) -> float:
            if native_is_categorical:
                try:
                    return float(native_y_values[[str(x) for x in native_x_values].index(str(value))])
                except (ValueError, IndexError, TypeError):
                    return 0.0
            try:
                native_x_numeric = np.asarray(native_x_values, dtype=float)
                native_y_numeric = np.asarray(native_y_values, dtype=float)
                return float(np.interp(float(value), native_x_numeric, native_y_numeric))
            except (TypeError, ValueError):
                return 0.0

        if is_categorical:
            raw_x_values = self._get_chart_x_values(
                feature_name,
                treat_as_categorical=True,
            )
            shape_values = []
            effective_x_values: List[str] = []
            for raw_value in raw_x_values:
                sample_data = baseline.copy()
                if feature_name in self.num_features:
                    indicator_name = self._missing_indicator_feature_name(feature_name)
                    if raw_value == MISSING_CATEGORY_VALUE:
                        if not indicator_name:
                            continue
                        sample_data[feature_name] = float(
                            self.numeric_missing_placeholder_values.get(
                                feature_name,
                                self.feature_schema_map.get(
                                    feature_name,
                                    {},
                                ).get("default_value", 0.0),
                            )
                        )
                        sample_data[indicator_name] = MISSING_CATEGORY_VALUE
                    else:
                        try:
                            sample_data[feature_name] = float(raw_value)
                        except (TypeError, ValueError):
                            continue
                        if indicator_name:
                            sample_data[indicator_name] = "Observed"
                else:
                    sample_data[feature_name] = raw_value
                shape_values.append(native_effect_at(raw_value))
                effective_x_values.append(raw_value)

            if not effective_x_values:
                is_categorical = False

            if is_categorical:
                x_tick_labels = [
                    str(categorical_value_labels.get(raw_value, raw_value))
                    for raw_value in effective_x_values
                ]

                result = {
                    "feature_name": feature_name,
                    "x_values": effective_x_values,
                    "x_tick_labels": x_tick_labels,
                    "y_values": shape_values,
                    "feature_type": "categorical",
                    "chart_config": chart_setting,
                }
                if include_distribution:
                    result["distribution"] = self._build_shape_function_distribution(
                        feature_name,
                        "categorical",
                        effective_x_values,
                        x_tick_labels=x_tick_labels,
                    )
                return result

        if feature_name in self.cat_features:
            numeric_pairs = self._get_numeric_chart_values_for_categorical_feature(
                feature_name,
            )
            if numeric_pairs:
                x_values_numeric: List[float] = []
                shape_values = []
                for raw_value, numeric_x in numeric_pairs:
                    sample_data = baseline.copy()
                    sample_data[feature_name] = raw_value
                    shape_values.append(native_effect_at(raw_value))
                    x_values_numeric.append(float(numeric_x))

                missing_bucket = (
                    self._build_numeric_chart_missing_bucket(
                        feature_name,
                        baseline,
                    )
                    if self.show_missing_bars
                    else None
                )

                result = {
                    "feature_name": feature_name,
                    "x_values": x_values_numeric,
                    "y_values": shape_values,
                    "feature_type": "numeric",
                    "x_tick_labels": self._build_numeric_chart_tick_labels(
                        x_values_numeric,
                        numeric_tick_labels,
                    ),
                    "chart_config": chart_setting,
                    "missing_bucket": missing_bucket,
                }
                if include_distribution:
                    result["distribution"] = self._build_shape_function_distribution(
                        feature_name,
                        "numeric",
                        x_values_numeric,
                    )
                return result

        x_values_numeric = self._get_numeric_chart_x_values(feature_name)
        if not x_values_numeric:
            raise ValueError(f"No observed numeric values available for '{feature_name}'")

        shape_values = []
        for x_val in x_values_numeric:
            sample_data = baseline.copy()
            sample_data[feature_name] = float(x_val)
            indicator_name = self._missing_indicator_feature_name(feature_name)
            if indicator_name:
                sample_data[indicator_name] = "Observed"
            shape_values.append(native_effect_at(x_val))

        missing_bucket = (
            self._build_numeric_chart_missing_bucket(
                feature_name,
                baseline,
            )
            if self.show_missing_bars
            else None
        )

        result = {
            "feature_name": feature_name,
            "x_values": x_values_numeric,
            "y_values": shape_values,
            "feature_type": "numeric",
            "x_tick_labels": self._build_numeric_chart_tick_labels(
                x_values_numeric,
                numeric_tick_labels,
            ),
            "chart_config": chart_setting,
            "missing_bucket": missing_bucket,
        }
        if include_distribution:
            result["distribution"] = self._build_shape_function_distribution(
                feature_name,
                "numeric",
                x_values_numeric,
            )
        return result

    def get_predictions_vs_actual(self) -> Dict[str, List[float]]:
        """Get predictions vs actual values for visualization."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        self._require_analytics_data()

        y_pred = self.model.predict(self.X_test)
        y_actual = self.y_test.values.flatten().tolist()
        return {
            "predicted": [float(value) for value in y_pred],
            "actual": y_actual,
        }

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the loaded dataset."""
        if self.df is None:
            raise ValueError("Data not loaded yet")

        target_values = pd.to_numeric(self.df[self.target_column], errors="coerce")
        return {
            "total_records": len(self.df),
            "features": self.selected_feature_columns,
            "numeric_features": [
                feature["name"]
                for feature in self.feature_schema
                if feature.get("feature_type") == "numeric"
            ],
            "categorical_features": [
                feature["name"]
                for feature in self.feature_schema
                if feature.get("feature_type") == "categorical"
            ],
            "target_column": self.target_column,
            "target_stats": {
                "mean": float(target_values.mean()),
                "std": float(target_values.std()),
                "min": float(target_values.min()),
                "max": float(target_values.max()),
            },
        }

    def get_feature_distributions(self) -> Dict[str, Any]:
        """Get feature distributions for visualization."""
        if self.X_train is None:
            raise ValueError("Data not loaded yet")

        distributions: Dict[str, Any] = {}
        numeric_features = [
            feature["name"]
            for feature in self.feature_schema
            if feature.get("feature_type") == "numeric"
        ]
        categorical_features = [
            feature["name"]
            for feature in self.feature_schema
            if feature.get("feature_type") == "categorical"
        ]

        for feature in numeric_features:
            values = pd.to_numeric(
                self._get_public_training_series(feature),
                errors="coerce",
            ).dropna().tolist()
            distributions[feature] = {
                "type": "numeric",
                "values": values[:1000],
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
            }

        for feature in categorical_features:
            raw_series = self._get_public_training_series(feature)
            value_counts = (
                raw_series.astype(object)
                .where(raw_series.notna(), MISSING_CATEGORY_VALUE)
                .astype(str)
                .value_counts()
                .to_dict()
            )
            distributions[feature] = {
                "type": "categorical",
                "counts": {str(key): int(value) for key, value in value_counts.items()},
            }

        return distributions

    def get_hourly_pattern(self) -> Dict[str, Any]:
        """Get hourly target pattern when standard bike columns exist."""
        if self.df is None:
            raise ValueError("Data not loaded yet")

        if "hr" not in self.df.columns or self.target_column not in self.df.columns:
            return {
                "available": False,
                "hours": [],
                "avg_rentals": [],
                "message": "Hourly pattern is not available for this dataset.",
            }

        hourly_avg = self.df.groupby("hr")[self.target_column].mean()
        return {
            "available": True,
            "hours": hourly_avg.index.tolist(),
            "avg_rentals": hourly_avg.values.tolist(),
        }
