"""Feature schema and chart display settings for active model pages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_processing import MISSING_CATEGORY_VALUE
from .ml_shared import _json_safe


class ChartServiceMixin:
    @staticmethod
    def _stringify_chart_value(value: Any) -> str:
        """Convert raw values into stable, human-readable chart keys."""
        if value is None:
            return "Unknown"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                return str(numeric)
            rounded = round(numeric)
            if abs(numeric - rounded) < 1e-9:
                return str(int(rounded))
            return format(numeric, ".12g")
        return str(value)

    @staticmethod
    def _is_integer_like_numeric(values: pd.Series) -> bool:
        if values.empty:
            return False
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return False
        arr = numeric.to_numpy(dtype=float)
        return bool(np.all(np.isclose(arr, np.round(arr))))

    def _feature_can_be_categorical(self, feature_name: str) -> bool:
        """Whether a feature can safely be treated as categorical for charts."""
        if feature_name in self.cat_features:
            return True
        series = pd.to_numeric(self._get_public_training_series(feature_name), errors="coerce").dropna()
        if series.empty or not self._is_integer_like_numeric(series):
            return False

        unique_count = int(series.nunique(dropna=True))
        return unique_count <= 13

    def _feature_can_be_numeric(self, feature_name: str) -> bool:
        """Whether a feature can safely be treated as numeric for charts."""
        if feature_name in self.num_features:
            return True
        series = self._get_public_training_series(feature_name)
        non_null = series.dropna()
        if non_null.empty:
            return False

        numeric = pd.to_numeric(non_null, errors="coerce")
        return bool(numeric.notna().all())

    @classmethod
    def _sanitize_chart_label_map(cls, raw_labels: Any) -> Dict[str, str]:
        if not isinstance(raw_labels, dict):
            return {}
        sanitized: Dict[str, str] = {}
        for raw_key, raw_label in raw_labels.items():
            label_str = str(raw_label).strip()
            if not label_str:
                continue
            sanitized[cls._stringify_chart_value(raw_key)] = label_str
        return sanitized

    @classmethod
    def _normalize_stored_feature_chart_setting(
        cls,
        raw_setting: Any,
    ) -> Dict[str, Any]:
        if not isinstance(raw_setting, dict):
            return {}
        display_title = str(raw_setting.get("display_title", "") or "").strip()
        x_axis_label = str(raw_setting.get("x_axis_label", "") or "").strip()
        categorical_value_labels = cls._sanitize_chart_label_map(
            raw_setting.get("categorical_value_labels", raw_setting.get("value_labels")),
        )
        numeric_tick_labels = cls._sanitize_chart_label_map(
            raw_setting.get("numeric_tick_labels"),
        )
        return {
            "display_title": display_title,
            "x_axis_label": x_axis_label,
            "treat_as_categorical": bool(raw_setting.get("treat_as_categorical")),
            "treat_as_numeric": bool(raw_setting.get("treat_as_numeric")),
            "categorical_value_labels": categorical_value_labels,
            "numeric_tick_labels": numeric_tick_labels,
        }

    def _get_feature_chart_setting(self, feature_name: str) -> Dict[str, Any]:
        normalized = self._normalize_stored_feature_chart_setting(
            self.feature_chart_settings.get(feature_name, {}),
        )
        if not normalized:
            return {
                "display_title": "",
                "x_axis_label": "",
                "treat_as_categorical": False,
                "treat_as_numeric": False,
                "categorical_value_labels": {},
                "numeric_tick_labels": {},
            }
        return normalized

    def _get_feature_display_title(self, feature_name: str) -> str:
        stored = self._get_feature_chart_setting(feature_name)
        display_title = str(stored.get("display_title", "") or "").strip()
        return display_title or feature_name

    def _with_feature_display_name(
        self,
        feature_schema_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = _json_safe(feature_schema_entry)
        feature_name = str(enriched.get("name", "") or "").strip()
        if feature_name:
            enriched["display_name"] = self._get_feature_display_title(feature_name)
        return enriched

    def _get_feature_chart_mode(self, feature_name: str) -> str:
        """Return effective chart display mode ('numeric' or 'categorical')."""
        setting = self._get_feature_chart_setting(feature_name)
        base_feature_type = "categorical" if feature_name in self.cat_features else "numeric"

        if base_feature_type == "numeric":
            if bool(setting.get("treat_as_categorical")) and self._feature_can_be_categorical(feature_name):
                return "categorical"
            return "numeric"

        if bool(setting.get("treat_as_numeric")) and self._feature_can_be_numeric(feature_name):
            return "numeric"
        return "categorical"

    def _is_feature_categorical_for_chart(self, feature_name: str) -> bool:
        return self._get_feature_chart_mode(feature_name) == "categorical"

    def _get_numeric_chart_x_values(self, feature_name: str) -> List[float]:
        if feature_name in self.cat_features:
            return [
                float(numeric_x)
                for _, numeric_x in self._get_numeric_chart_values_for_categorical_feature(
                    feature_name,
                )
            ]

        public_numeric_series = pd.to_numeric(
            self._get_public_training_series(feature_name),
            errors="coerce",
        ).dropna()
        if public_numeric_series.empty:
            return []

        schema = self.feature_schema_map.get(feature_name, {}) or {}
        schema_min = schema.get("min_value")
        schema_max = schema.get("max_value")
        try:
            min_val = float(schema_min)
        except (TypeError, ValueError):
            min_val = float(public_numeric_series.min())
        try:
            max_val = float(schema_max)
        except (TypeError, ValueError):
            max_val = float(public_numeric_series.max())

        if not np.isfinite(min_val):
            min_val = float(public_numeric_series.min())
        if not np.isfinite(max_val):
            max_val = float(public_numeric_series.max())
        if max_val < min_val:
            min_val = float(public_numeric_series.min())
            max_val = float(public_numeric_series.max())

        if min_val == max_val:
            return [min_val]
        return np.linspace(min_val, max_val, 30).tolist()

    def _get_numeric_chart_label_values(self, feature_name: str) -> List[str]:
        numeric_x_values = self._get_numeric_chart_x_values(feature_name)
        if not numeric_x_values:
            return []

        min_val = min(numeric_x_values)
        max_val = max(numeric_x_values)
        start = int(np.ceil(min_val - 1e-9))
        end = int(np.floor(max_val + 1e-9))
        if end < start:
            return []
        return [str(value) for value in range(start, end + 1)]

    def _build_numeric_chart_tick_labels(
        self,
        x_values: List[float],
        numeric_tick_labels: Dict[str, str],
    ) -> Optional[List[str]]:
        labels: List[str] = []
        has_custom_labels = False
        for x_value in x_values:
            label = str(
                numeric_tick_labels.get(self._stringify_chart_value(x_value), "")
            ).strip()
            if label:
                has_custom_labels = True
            labels.append(label)
        return labels if has_custom_labels else None

    def get_chart_display_settings(self) -> Dict[str, Any]:
        return {
            "show_missing_bars": bool(self.show_missing_bars),
            "show_competence_levels": bool(self.show_competence_levels),
            "show_prediction_comparison_superadmin": bool(
                self.show_prediction_comparison_superadmin
            ),
            "show_prediction_comparison_users": bool(
                self.show_prediction_comparison_users
            ),
        }

    def update_chart_display_settings(
        self,
        show_missing_bars: Optional[bool] = None,
        show_competence_levels: Optional[bool] = None,
        show_prediction_comparison_superadmin: Optional[bool] = None,
        show_prediction_comparison_users: Optional[bool] = None,
    ) -> Dict[str, Any]:
        chart_changed = False
        if show_missing_bars is not None:
            chart_changed |= self.show_missing_bars != show_missing_bars
            self.show_missing_bars = show_missing_bars
        if show_competence_levels is not None:
            chart_changed |= self.show_competence_levels != show_competence_levels
            self.show_competence_levels = show_competence_levels
        if show_prediction_comparison_superadmin is not None:
            self.show_prediction_comparison_superadmin = (
                show_prediction_comparison_superadmin
            )
        if show_prediction_comparison_users is not None:
            self.show_prediction_comparison_users = show_prediction_comparison_users
        if chart_changed:
            # Chart modes change the shape-function snapshot and active offsets.
            self.original_shape_functions = {}
            self.shape_function_offsets = {}
        self._persist_active_dataset_metadata()
        return self.get_chart_display_settings()

    def _get_chart_x_values(
        self,
        feature_name: str,
        *,
        treat_as_categorical: Optional[bool] = None,
    ) -> List[str]:
        """Get raw x-axis values that should be used for a categorical chart."""
        series = self._get_public_training_series(feature_name)
        if series.empty and feature_name not in self.feature_schema_map:
            return []

        if treat_as_categorical is None:
            treat_as_categorical = self._is_feature_categorical_for_chart(feature_name)

        if not treat_as_categorical:
            return []

        if str(self.feature_schema_map.get(feature_name, {}).get("feature_type")) == "categorical":
            options = self.feature_schema_map.get(feature_name, {}).get("categorical_options", [])
            base_values = [self._stringify_chart_value(v) for v in options]
        else:
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            uniques = sorted(numeric_series.unique().tolist())
            base_values = [self._stringify_chart_value(v) for v in uniques]
            if self.show_missing_bars and self._get_public_missing_count(feature_name) > 0:
                base_values.append(MISSING_CATEGORY_VALUE)

        deduped: List[str] = []
        seen = set()
        for raw in base_values:
            if not self.show_missing_bars and raw == MISSING_CATEGORY_VALUE:
                continue
            if raw in seen:
                continue
            seen.add(raw)
            deduped.append(raw)

        # Keep intuitive ordering for integer-coded categories like month (1..12).
        numeric_pairs: List[Tuple[float, str]] = []
        for raw in deduped:
            try:
                numeric_pairs.append((float(raw), raw))
            except (TypeError, ValueError):
                numeric_pairs = []
                break
        if numeric_pairs:
            numeric_pairs.sort(key=lambda item: item[0])
            deduped = [raw for _, raw in numeric_pairs]
        return deduped

    def _get_numeric_chart_values_for_categorical_feature(
        self,
        feature_name: str,
    ) -> List[Tuple[str, float]]:
        """Return (raw_value, numeric_x) pairs for categorical-as-numeric display."""
        raw_values = self._get_chart_x_values(
            feature_name,
            treat_as_categorical=True,
        )

        pairs: List[Tuple[str, float]] = []
        for raw in raw_values:
            try:
                numeric_val = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_val):
                continue
            pairs.append((str(raw), float(numeric_val)))

        pairs.sort(key=lambda item: item[1])
        return pairs

    def get_feature_chart_setting(self, feature_name: str) -> Dict[str, Any]:
        """Return effective chart settings for one feature."""
        if feature_name not in self._public_feature_names():
            raise ValueError(f"Unknown feature: {feature_name}")

        base_feature_type = str(
            self.feature_schema_map.get(feature_name, {}).get("feature_type", "numeric")
        )
        can_be_categorical = self._feature_can_be_categorical(feature_name)
        can_be_numeric = self._feature_can_be_numeric(feature_name)
        stored = self._get_feature_chart_setting(feature_name)
        treat_as_categorical = bool(stored.get("treat_as_categorical")) if base_feature_type == "numeric" else False
        treat_as_numeric = bool(stored.get("treat_as_numeric")) if base_feature_type == "categorical" else False

        if treat_as_categorical and not can_be_categorical:
            treat_as_categorical = False
        if treat_as_numeric and not can_be_numeric:
            treat_as_numeric = False

        chart_feature_type = "categorical"
        if base_feature_type == "numeric":
            chart_feature_type = "categorical" if treat_as_categorical else "numeric"
        else:
            chart_feature_type = "numeric" if treat_as_numeric else "categorical"

        is_categorical = chart_feature_type == "categorical"
        is_numeric = chart_feature_type == "numeric"
        allowed_categorical_values = set(
            self._get_chart_x_values(feature_name, treat_as_categorical=True)
        )
        available_categorical_values = (
            self._get_chart_x_values(feature_name, treat_as_categorical=True)
            if can_be_categorical or base_feature_type == "categorical"
            else []
        )
        available_numeric_values = (
            self._get_numeric_chart_label_values(feature_name)
            if can_be_numeric or base_feature_type == "numeric"
            else []
        )
        numeric_domain_values = (
            self._get_numeric_chart_x_values(feature_name)
            if can_be_numeric or base_feature_type == "numeric"
            else []
        )
        allowed_numeric_values = set(available_numeric_values)
        categorical_value_labels = {
            str(raw): str(label).strip()
            for raw, label in (stored.get("categorical_value_labels", {}) or {}).items()
            if str(label).strip()
            and (not allowed_categorical_values or str(raw) in allowed_categorical_values)
        }
        numeric_tick_labels = {
            str(raw): str(label).strip()
            for raw, label in (stored.get("numeric_tick_labels", {}) or {}).items()
            if str(label).strip()
            and (not allowed_numeric_values or str(raw) in allowed_numeric_values)
        }

        return {
            "feature_name": feature_name,
            "display_title": str(stored.get("display_title", "") or "").strip(),
            "x_axis_label": str(stored.get("x_axis_label", "") or "").strip(),
            "base_feature_type": base_feature_type,
            "chart_feature_type": chart_feature_type,
            "is_numeric": is_numeric,
            "is_categorical": is_categorical,
            "can_be_categorical": can_be_categorical,
            "can_be_numeric": can_be_numeric,
            "treat_as_categorical": treat_as_categorical,
            "treat_as_numeric": treat_as_numeric,
            "categorical_value_labels": categorical_value_labels,
            "numeric_tick_labels": numeric_tick_labels,
            "value_labels": categorical_value_labels,
            "available_categorical_values": available_categorical_values,
            "available_numeric_values": available_numeric_values,
            "numeric_domain_min": (
                float(min(numeric_domain_values)) if numeric_domain_values else None
            ),
            "numeric_domain_max": (
                float(max(numeric_domain_values)) if numeric_domain_values else None
            ),
        }

    def update_feature_chart_setting(
        self,
        feature_name: str,
        treat_as_categorical: bool,
        treat_as_numeric: bool = False,
        display_title: Optional[str] = None,
        x_axis_label: Optional[str] = None,
        categorical_value_labels: Optional[Dict[str, str]] = None,
        numeric_tick_labels: Optional[Dict[str, str]] = None,
        value_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Persist superadmin chart settings for one feature."""
        if self.X_train is None:
            raise ValueError("Data not loaded yet")
        if feature_name not in self.feature_names:
            raise ValueError(f"Unknown feature: {feature_name}")

        base_feature_type = "categorical" if feature_name in self.cat_features else "numeric"
        can_be_categorical = self._feature_can_be_categorical(feature_name)
        can_be_numeric = self._feature_can_be_numeric(feature_name)
        current = self._get_feature_chart_setting(feature_name)

        normalized_treat_as_categorical = False
        normalized_treat_as_numeric = False
        if base_feature_type == "numeric":
            normalized_treat_as_categorical = bool(treat_as_categorical)
            if normalized_treat_as_categorical and not can_be_categorical:
                raise ValueError(
                    f"Feature '{feature_name}' cannot be treated as categorical (too many distinct values)"
                )
        else:
            normalized_treat_as_numeric = bool(treat_as_numeric)
            if normalized_treat_as_numeric and not can_be_numeric:
                raise ValueError(
                    f"Feature '{feature_name}' cannot be treated as numeric (values are not numeric)"
                )

        effective_categorical = (
            base_feature_type == "categorical" and not normalized_treat_as_numeric
        ) or (
            base_feature_type == "numeric" and normalized_treat_as_categorical
        )
        allowed_categorical_values = set(
            self._get_chart_x_values(feature_name, treat_as_categorical=True)
        )
        allowed_numeric_values = {
            self._stringify_chart_value(value)
            for value in self._get_numeric_chart_label_values(feature_name)
        }

        source_categorical_labels = (
            current.get("categorical_value_labels", {})
            if categorical_value_labels is None and value_labels is None
            else (
                categorical_value_labels
                if categorical_value_labels is not None
                else (value_labels or {})
            )
        )
        sanitized_categorical_labels: Dict[str, str] = {}
        for raw_key, raw_label in (source_categorical_labels or {}).items():
            key = self._stringify_chart_value(raw_key)
            label = str(raw_label).strip()
            if not label:
                continue
            if allowed_categorical_values and key not in allowed_categorical_values:
                continue
            sanitized_categorical_labels[key] = label

        source_numeric_labels = (
            current.get("numeric_tick_labels", {})
            if numeric_tick_labels is None
            else (numeric_tick_labels or {})
        )
        sanitized_numeric_labels: Dict[str, str] = {}
        for raw_key, raw_label in (source_numeric_labels or {}).items():
            key = self._stringify_chart_value(raw_key)
            label = str(raw_label).strip()
            if not label:
                continue
            if allowed_numeric_values and key not in allowed_numeric_values:
                continue
            sanitized_numeric_labels[key] = label

        sanitized_display_title = (
            str(current.get("display_title", "") or "").strip()
            if display_title is None
            else str(display_title or "").strip()
        )
        sanitized_x_axis_label = (
            str(current.get("x_axis_label", "") or "").strip()
            if x_axis_label is None
            else str(x_axis_label or "").strip()
        )

        normalized_setting = {
            "display_title": sanitized_display_title,
            "x_axis_label": sanitized_x_axis_label,
            "treat_as_categorical": normalized_treat_as_categorical,
            "treat_as_numeric": normalized_treat_as_numeric,
            "categorical_value_labels": sanitized_categorical_labels,
            "numeric_tick_labels": sanitized_numeric_labels,
        }

        if (
            not normalized_setting["display_title"]
            and not normalized_setting["x_axis_label"]
            and not normalized_setting["treat_as_categorical"]
            and not normalized_setting["treat_as_numeric"]
            and not normalized_setting["categorical_value_labels"]
            and not normalized_setting["numeric_tick_labels"]
        ):
            self.feature_chart_settings.pop(feature_name, None)
        else:
            self.feature_chart_settings[feature_name] = normalized_setting

        # Clear runtime chart snapshots/offsets so the next fetch reflects the new mode.
        self.original_shape_functions = {}
        self.shape_function_offsets = {}
        self._persist_active_dataset_metadata()
        return self.get_feature_chart_setting(feature_name)

    def _build_feature_schema_for(
        self,
        feature_names: List[str],
        num_features: List[str],
        X_all: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        schema: List[Dict[str, Any]] = []
        for feature_name in feature_names:
            if feature_name in num_features:
                numeric_series = pd.to_numeric(X_all[feature_name], errors="coerce")
                valid_numeric_series = numeric_series.dropna()

                if valid_numeric_series.empty:
                    min_val = 0.0
                    max_val = 0.0
                    default_val = 0.0
                else:
                    min_val = float(valid_numeric_series.min())
                    max_val = float(valid_numeric_series.max())
                    default_val = float(valid_numeric_series.median())

                    if default_val < min_val:
                        default_val = min_val
                    elif default_val > max_val:
                        default_val = max_val

                default_val = round(default_val, 2)

                schema.append(
                    {
                        "name": feature_name,
                        "feature_type": "numeric",
                        "default_value": default_val,
                        "min_value": min_val,
                        "max_value": max_val,
                    }
                )
            else:
                cat_series = X_all[feature_name].astype(str)
                unique_values = [str(v) for v in pd.unique(cat_series)]
                unique_values = sorted(unique_values)
                if not unique_values:
                    unique_values = ["Unknown"]
                value_counts = cat_series.value_counts()
                default_val = str(value_counts.index[0]) if not value_counts.empty else unique_values[0]
                if default_val not in unique_values:
                    default_val = unique_values[0]

                schema.append(
                    {
                        "name": feature_name,
                        "feature_type": "categorical",
                        "default_value": default_val,
                        "categorical_options": unique_values,
                    }
                )

        return schema

    def _build_feature_schema(self, X_all: pd.DataFrame) -> List[Dict[str, Any]]:
        return self._build_feature_schema_for(self.feature_names, self.num_features, X_all)

    def _build_public_feature_schema(
        self,
        selected_features: List[str],
        public_numeric_features: List[str],
        public_frame: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Build the public-facing schema from the original selected features only."""
        schema: List[Dict[str, Any]] = []
        public_numeric_set = set(public_numeric_features)

        for feature_name in selected_features:
            if feature_name in public_numeric_set:
                numeric_series = pd.to_numeric(public_frame[feature_name], errors="coerce")
                valid_numeric_series = numeric_series.dropna()

                if valid_numeric_series.empty:
                    min_val = 0.0
                    max_val = 0.0
                    default_val = 0.0
                else:
                    min_val = float(valid_numeric_series.min())
                    max_val = float(valid_numeric_series.max())
                    default_val = float(valid_numeric_series.median())

                schema.append(
                    {
                        "name": feature_name,
                        "feature_type": "numeric",
                        "default_value": round(float(default_val), 2),
                        "min_value": min_val,
                        "max_value": max_val,
                    }
                )
                continue

            cat_series = (
                public_frame[feature_name]
                .astype(object)
                .where(public_frame[feature_name].notna(), MISSING_CATEGORY_VALUE)
                .astype(str)
            )
            unique_values = [str(v) for v in pd.unique(cat_series)]
            unique_values = sorted(unique_values)
            if not unique_values:
                unique_values = [MISSING_CATEGORY_VALUE]

            observed_counts = cat_series[cat_series != MISSING_CATEGORY_VALUE].value_counts()
            if not observed_counts.empty:
                default_val = str(observed_counts.index[0])
            else:
                default_val = unique_values[0]

            if default_val not in unique_values:
                default_val = unique_values[0]

            schema.append(
                {
                    "name": feature_name,
                    "feature_type": "categorical",
                    "default_value": default_val,
                    "categorical_options": unique_values,
                }
            )

        return schema
