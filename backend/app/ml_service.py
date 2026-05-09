"""Machine learning service for persisted datasets and model versions."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from igann import IGANN
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .data_processing import (
    build_data_summary,
    build_dataset_schema,
    build_hourly_pattern,
    inspect_csv_bytes,
    prepare_training_data,
    read_csv_bytes,
)


DATA_FILE_NAME = "bike.csv"


def _candidate_data_files() -> List[Path]:
    """Return likely dataset locations across local and Azure runtime layouts."""
    app_dir = Path(__file__).resolve().parent
    backend_dir = app_dir.parent
    repo_root = backend_dir.parent
    cwd = Path.cwd().resolve()

    candidates: List[Path] = []
    configured = os.getenv("DATA_FILE_PATH", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            repo_root / DATA_FILE_NAME,
            backend_dir / DATA_FILE_NAME,
            cwd / DATA_FILE_NAME,
            cwd / "backend" / DATA_FILE_NAME,
        ]
    )

    seen = set()
    deduped: List[Path] = []
    for path in candidates:
        normalized = path.resolve() if not path.is_absolute() else path
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _resolve_data_file(csv_path: Optional[str] = None) -> Path:
    """Resolve the default dataset path and raise a clear error if not found."""
    if csv_path:
        explicit = Path(csv_path).expanduser()
        resolved = explicit.resolve() if not explicit.is_absolute() else explicit
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(f"Dataset not found at explicit path: {resolved}")

    checked_paths: List[Path] = []
    for candidate in _candidate_data_files():
        checked_paths.append(candidate)
        if candidate.is_file():
            return candidate

    checked = "\n".join(f"- {path}" for path in checked_paths)
    raise FileNotFoundError(
        "Dataset not found. Checked the following locations:\n"
        f"{checked}\n"
        "Set DATA_FILE_PATH to a valid bike.csv location if needed."
    )


class MLService:
    """Service for training, caching, and evaluating IGANN model versions."""

    def __init__(self) -> None:
        self._artifact_cache: Dict[int, Dict[str, Any]] = {}

    # ==================== Dataset Helpers ====================

    def inspect_dataset_upload(self, csv_bytes: bytes) -> Dict[str, Any]:
        return inspect_csv_bytes(csv_bytes)

    def build_dataset_schema(self, csv_bytes: bytes, target_column: str) -> Dict[str, Any]:
        df = read_csv_bytes(csv_bytes)
        return build_dataset_schema(df, target_column)

    def build_default_dataset_payload(self) -> Dict[str, Any]:
        path = _resolve_data_file()
        csv_bytes = path.read_bytes()
        schema = self.build_dataset_schema(csv_bytes, "cnt")
        return {
            "display_name": "Sample Bike Dataset",
            "original_filename": path.name,
            "target_column": "cnt",
            "schema": schema,
            "csv_data": csv_bytes,
        }

    def _read_dataset_df(self, dataset_record: Dict[str, Any]) -> pd.DataFrame:
        return read_csv_bytes(dataset_record["csv_data"])

    def get_dataset_summary(self, dataset_record: Dict[str, Any]) -> Dict[str, Any]:
        df = self._read_dataset_df(dataset_record)
        return build_data_summary(df, dataset_record["schema"])

    def get_hourly_pattern(self, dataset_record: Dict[str, Any]) -> Dict[str, Any]:
        df = self._read_dataset_df(dataset_record)
        return build_hourly_pattern(df, dataset_record["schema"])

    # ==================== Training and Artifact Persistence ====================

    def train_model(
        self,
        dataset_record: Dict[str, Any],
        n_estimators: int = 100,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        df = self._read_dataset_df(dataset_record)
        dataset_schema = dataset_record["schema"]
        X_train, X_test, y_train, y_test, fill_values, _ = prepare_training_data(
            df,
            dataset_schema,
            test_size=test_size,
            random_state=random_state,
        )

        model = IGANN(
            task="regression",
            n_estimators=n_estimators,
            verbose=0,
            scale_y=True,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "model_type": "IGANN",
        }

        artifact: Dict[str, Any] = {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "schema": dataset_schema,
            "fill_values": fill_values,
            "feature_names": list(X_train.columns),
            "metrics": metrics,
        }

        original_shape_functions, original_shape_functions_map = self._extract_all_shape_functions(artifact)
        artifact["original_shape_functions"] = original_shape_functions
        artifact["original_shape_functions_map"] = original_shape_functions_map

        return {
            "metrics": metrics,
            "artifact": artifact,
            "artifact_blob": pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "schema_snapshot": dataset_schema,
            "training_params": {
                "n_estimators": n_estimators,
                "test_size": test_size,
                "random_state": random_state,
            },
        }

    def prime_artifact_cache(self, model_version_id: int, artifact: Dict[str, Any]) -> None:
        self._artifact_cache[model_version_id] = artifact

    def load_artifact(self, model_version_record: Dict[str, Any]) -> Dict[str, Any]:
        model_version_id = model_version_record["id"]
        if model_version_id in self._artifact_cache:
            return self._artifact_cache[model_version_id]

        artifact_blob = model_version_record.get("artifact_blob")
        if artifact_blob is None:
            raise ValueError("Model version artifact is missing")

        artifact = pickle.loads(artifact_blob)
        if "original_shape_functions_map" not in artifact:
            artifact["original_shape_functions_map"] = {
                sf["feature_name"]: sf for sf in artifact.get("original_shape_functions", [])
            }

        self._artifact_cache[model_version_id] = artifact
        return artifact

    # ==================== Shape Function Extraction ====================

    def _extract_all_shape_functions(
        self,
        artifact: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        feature_names = artifact["feature_names"]
        shape_functions: List[Dict[str, Any]] = []
        shape_map: Dict[str, Dict[str, Any]] = {}

        for feature_idx, feature_name in enumerate(feature_names):
            shape_data = self._extract_shape_function(artifact, feature_name, feature_idx)
            shape_functions.append(shape_data)
            shape_map[feature_name] = shape_data

        return shape_functions, shape_map

    def _extract_shape_function(
        self,
        artifact: Dict[str, Any],
        feature_name: str,
        feature_idx: int,
    ) -> Dict[str, Any]:
        X_train = artifact["X_train"]
        model = artifact["model"]
        schema = artifact["schema"]
        numeric_features = schema["numeric_features"]
        categorical_features = schema["categorical_features"]
        is_categorical = feature_name in categorical_features

        baseline: Dict[str, Any] = {}
        for num_feat in numeric_features:
            baseline[num_feat] = float(pd.to_numeric(X_train[num_feat], errors="coerce").mean())
        for cat_feat in categorical_features:
            mode = X_train[cat_feat].astype(str).mode()
            baseline[cat_feat] = str(mode.iloc[0]) if not mode.empty else ""

        if is_categorical:
            unique_values = [str(v) for v in X_train[feature_name].astype(str).unique().tolist()]
            shape_values: List[float] = []
            for value in unique_values:
                sample_data = baseline.copy()
                sample_data[feature_name] = value
                sample = pd.DataFrame([sample_data])[schema["feature_names"]]
                if categorical_features:
                    sample = sample.astype({cat: "object" for cat in categorical_features})
                for num_feat in numeric_features:
                    sample[num_feat] = pd.to_numeric(sample[num_feat], errors="coerce").astype(float)

                pred = model.predict(sample)
                shape_values.append(float(pred[0]) if hasattr(pred, "__iter__") else float(pred))

            mean_value = np.mean(shape_values) if shape_values else 0.0
            shape_values = [float(value - mean_value) for value in shape_values]
            return {
                "feature_name": feature_name,
                "x_values": unique_values,
                "y_values": shape_values,
                "feature_type": "categorical",
            }

        min_value = float(pd.to_numeric(X_train[feature_name], errors="coerce").min())
        max_value = float(pd.to_numeric(X_train[feature_name], errors="coerce").max())
        if np.isclose(min_value, max_value):
            x_range = np.array([min_value], dtype=float)
        else:
            x_range = np.linspace(min_value, max_value, 30)

        shape_values = []
        for x_value in x_range:
            sample_data = baseline.copy()
            sample_data[feature_name] = float(x_value)
            sample = pd.DataFrame([sample_data])[schema["feature_names"]]
            if categorical_features:
                sample = sample.astype({cat: "object" for cat in categorical_features})
            for num_feat in numeric_features:
                sample[num_feat] = pd.to_numeric(sample[num_feat], errors="coerce").astype(float)

            pred = model.predict(sample)
            shape_values.append(float(pred[0]) if hasattr(pred, "__iter__") else float(pred))

        mean_value = np.mean(shape_values) if shape_values else 0.0
        shape_values = [float(value - mean_value) for value in shape_values]
        return {
            "feature_name": feature_name,
            "x_values": x_range.tolist(),
            "y_values": shape_values,
            "feature_type": "numeric",
        }

    # ==================== Prediction Helpers ====================

    def _build_input_df(self, artifact: Dict[str, Any], input_features: Dict[str, Any]) -> pd.DataFrame:
        schema = artifact["schema"]
        fill_values = artifact.get("fill_values", {"numeric": {}, "categorical": {}})
        payload: Dict[str, Any] = {}

        for field in schema["prediction_fields"]:
            name = field["name"]
            feature_type = field["feature_type"]
            value = input_features.get(name, field.get("default"))

            if feature_type == "numeric":
                if value in (None, ""):
                    value = fill_values["numeric"].get(name, field.get("default", 0.0))
                numeric_value = float(value)
                if field.get("is_integer"):
                    numeric_value = float(int(round(numeric_value)))
                payload[name] = numeric_value
            else:
                if value in (None, ""):
                    value = fill_values["categorical"].get(name, field.get("default", ""))
                payload[name] = str(value)

        df = pd.DataFrame([payload])[schema["feature_names"]]
        for feature_name in schema["numeric_features"]:
            fill_value = float(fill_values["numeric"].get(feature_name, 0.0))
            df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce").fillna(fill_value).astype(float)
        if schema["categorical_features"]:
            for feature_name in schema["categorical_features"]:
                fill_value = str(fill_values["categorical"].get(feature_name, ""))
                df[feature_name] = df[feature_name].fillna(fill_value).astype(str)
            df = df.astype({feature_name: "object" for feature_name in schema["categorical_features"]})
        return df

    def _get_offset_for_value(
        self,
        artifact: Dict[str, Any],
        feature_name: str,
        value: Any,
        offsets: Dict[str, Dict[Any, float]],
    ) -> float:
        feature_offsets = offsets.get(feature_name)
        if not feature_offsets:
            return 0.0

        schema = artifact["schema"]
        if feature_name in schema["categorical_features"]:
            return float(feature_offsets.get(str(value), 0.0))

        original = artifact["original_shape_functions_map"].get(feature_name, {})
        original_x = original.get("x_values", [])
        if not original_x:
            return 0.0

        try:
            indices = sorted(int(key) if isinstance(key, str) else key for key in feature_offsets.keys())
            offset_values = [float(feature_offsets.get(idx, feature_offsets.get(str(idx), 0.0))) for idx in indices]
            x_positions = [float(original_x[idx]) for idx in indices]
        except (TypeError, ValueError, IndexError):
            return 0.0

        if len(indices) == 1:
            return offset_values[0]

        value_float = float(value)
        if value_float <= x_positions[0]:
            return offset_values[0]
        if value_float >= x_positions[-1]:
            return offset_values[-1]

        for idx in range(len(x_positions) - 1):
            if x_positions[idx] <= value_float <= x_positions[idx + 1]:
                span = x_positions[idx + 1] - x_positions[idx]
                if np.isclose(span, 0.0):
                    return offset_values[idx]
                t = (value_float - x_positions[idx]) / span
                return offset_values[idx] + t * (offset_values[idx + 1] - offset_values[idx])
        return 0.0

    def _predict_with_offsets_df(
        self,
        artifact: Dict[str, Any],
        X: pd.DataFrame,
        offsets: Dict[str, Dict[Any, float]],
    ) -> np.ndarray:
        base_predictions = artifact["model"].predict(X)
        total_offsets = np.zeros(len(X))

        for feature_name, feature_offsets in offsets.items():
            if feature_name not in X.columns or not feature_offsets:
                continue
            for row_idx, value in enumerate(X[feature_name].values):
                total_offsets[row_idx] += self._get_offset_for_value(artifact, feature_name, value, offsets)

        return base_predictions + total_offsets

    def predict(
        self,
        model_version_record: Dict[str, Any],
        input_features: Dict[str, Any],
        offsets: Optional[Dict[str, Dict[Any, float]]] = None,
    ) -> float:
        artifact = self.load_artifact(model_version_record)
        input_df = self._build_input_df(artifact, input_features)
        base_prediction = float(artifact["model"].predict(input_df)[0])
        total_offset = 0.0
        for feature_name in input_df.columns:
            total_offset += self._get_offset_for_value(artifact, feature_name, input_df.iloc[0][feature_name], offsets or {})
        return base_prediction + total_offset

    def evaluate_model(self, model_version_record: Dict[str, Any]) -> Dict[str, float]:
        return dict(model_version_record["metrics"])

    def get_shape_functions(self, model_version_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        artifact = self.load_artifact(model_version_record)
        return artifact["original_shape_functions"]

    def get_predictions_vs_actual(self, model_version_record: Dict[str, Any]) -> Dict[str, List[float]]:
        artifact = self.load_artifact(model_version_record)
        predictions = artifact["model"].predict(artifact["X_test"])
        actual_values = artifact["y_test"].values.flatten().tolist()
        return {
            "predicted": [float(value) for value in predictions],
            "actual": [float(value) for value in actual_values],
        }

    def get_predictions_comparison(
        self,
        model_version_record: Dict[str, Any],
        offsets: Optional[Dict[str, Dict[Any, float]]] = None,
    ) -> Dict[str, Any]:
        artifact = self.load_artifact(model_version_record)
        offsets = offsets or {}

        original_predictions = artifact["model"].predict(artifact["X_test"])
        interactive_predictions = self._predict_with_offsets_df(artifact, artifact["X_test"], offsets)
        actual_values = artifact["y_test"].values.flatten()

        original_rmse = root_mean_squared_error(actual_values, original_predictions)
        original_mae = mean_absolute_error(actual_values, original_predictions)
        interactive_rmse = root_mean_squared_error(actual_values, interactive_predictions)
        interactive_mae = mean_absolute_error(actual_values, interactive_predictions)

        return {
            "original_predictions": [float(value) for value in original_predictions],
            "interactive_predictions": [float(value) for value in interactive_predictions],
            "actual_values": [float(value) for value in actual_values.tolist()],
            "metrics": {
                "original_rmse": float(original_rmse),
                "original_mae": float(original_mae),
                "interactive_rmse": float(interactive_rmse),
                "interactive_mae": float(interactive_mae),
            },
        }

    # ==================== Edit Conversion Helpers ====================

    def convert_edits_for_storage(
        self,
        model_version_record: Dict[str, Any],
        edited_shape_functions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        artifact = self.load_artifact(model_version_record)
        original_map = artifact["original_shape_functions_map"]
        result: List[Dict[str, Any]] = []

        for edited_sf in edited_shape_functions:
            feature_name = edited_sf["feature_name"]
            feature_type = edited_sf["feature_type"]
            original = original_map.get(feature_name)
            if not original:
                continue

            original_x = original["x_values"]
            original_y = original["y_values"]
            converted_points: List[Dict[str, Any]] = []

            for edited_point in edited_sf.get("edited_points", []):
                x_value = edited_point["x_value"]
                new_y = float(edited_point["y_value"])
                weight = float(edited_point.get("weight", 0.5))
                message = str(edited_point.get("message", ""))

                if feature_type == "categorical":
                    x_key = str(x_value)
                    if x_key in original_x:
                        idx = original_x.index(x_key)
                        offset = new_y - float(original_y[idx])
                        converted_points.append(
                            {
                                "x_value": x_key,
                                "y_value": offset,
                                "weight": weight,
                                "message": message,
                            }
                        )
                else:
                    x_float = float(x_value)
                    closest_idx = int(np.argmin(np.abs(np.array(original_x, dtype=float) - x_float)))
                    offset = new_y - float(original_y[closest_idx])
                    converted_points.append(
                        {
                            "x_value": closest_idx,
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
        model_version_record: Dict[str, Any],
        storage_edits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        artifact = self.load_artifact(model_version_record)
        original_map = artifact["original_shape_functions_map"]
        result: List[Dict[str, Any]] = []

        for stored_sf in storage_edits:
            feature_name = stored_sf["feature_name"]
            feature_type = stored_sf["feature_type"]
            original = original_map.get(feature_name)
            if not original:
                continue

            original_x = original["x_values"]
            original_y = original["y_values"]
            display_points: List[Dict[str, Any]] = []

            for stored_point in stored_sf.get("edited_points", []):
                offset = float(stored_point["y_value"])
                if feature_type == "categorical":
                    x_key = str(stored_point["x_value"])
                    if x_key in original_x:
                        idx = original_x.index(x_key)
                        display_points.append(
                            {
                                "x_value": x_key,
                                "y_value": float(original_y[idx]) + offset,
                                "weight": float(stored_point.get("weight", 0.5)),
                                "message": str(stored_point.get("message", "")),
                            }
                        )
                else:
                    try:
                        idx = int(stored_point["x_value"])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(original_x):
                        display_points.append(
                            {
                                "x_value": float(original_x[idx]),
                                "y_value": float(original_y[idx]) + offset,
                                "weight": float(stored_point.get("weight", 0.5)),
                                "message": str(stored_point.get("message", "")),
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

    def build_offsets_from_storage_edits(
        self,
        storage_edits: List[Dict[str, Any]],
    ) -> Dict[str, Dict[Any, float]]:
        offsets: Dict[str, Dict[Any, float]] = {}
        for sf in storage_edits:
            feature_name = sf["feature_name"]
            feature_type = sf["feature_type"]
            offsets[feature_name] = {}
            for point in sf.get("edited_points", []):
                x_value = point["x_value"]
                offset = float(point["y_value"])
                if feature_type == "categorical":
                    offsets[feature_name][str(x_value)] = offset
                else:
                    try:
                        offsets[feature_name][int(x_value)] = offset
                    except (TypeError, ValueError):
                        continue
        return offsets

    def merge_display_edits(
        self,
        saved_display_edits: List[Dict[str, Any]],
        overlay_display_edits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        feature_types: Dict[str, str] = {}

        for source in (saved_display_edits, overlay_display_edits):
            for sf in source:
                feature_name = sf["feature_name"]
                feature_types[feature_name] = sf["feature_type"]
                for point in sf.get("edited_points", []):
                    merged[(feature_name, str(point["x_value"]))] = {
                        "x_value": point["x_value"],
                        "y_value": point["y_value"],
                        "weight": point.get("weight", 0.5),
                        "message": point.get("message", ""),
                    }

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for (feature_name, _), point in merged.items():
            grouped.setdefault(feature_name, []).append(point)

        result: List[Dict[str, Any]] = []
        for feature_name, points in grouped.items():
            result.append(
                {
                    "feature_name": feature_name,
                    "feature_type": feature_types.get(feature_name, "numeric"),
                    "edited_points": points,
                }
            )
        return result

    def apply_offsets_to_shape_functions(
        self,
        model_version_record: Dict[str, Any],
        offsets: Dict[str, Dict[Any, float]],
    ) -> List[Dict[str, Any]]:
        artifact = self.load_artifact(model_version_record)
        result: List[Dict[str, Any]] = []

        for original in artifact["original_shape_functions"]:
            feature_name = original["feature_name"]
            feature_type = original["feature_type"]
            modified_y_values: List[float] = []
            for idx, (x_value, y_value) in enumerate(zip(original["x_values"], original["y_values"])):
                offset = 0.0
                if feature_name in offsets:
                    if feature_type == "categorical":
                        offset = float(offsets[feature_name].get(str(x_value), 0.0))
                    else:
                        offset = float(offsets[feature_name].get(idx, offsets[feature_name].get(str(idx), 0.0)))
                modified_y_values.append(float(y_value) + offset)

            result.append(
                {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "x_values": original["x_values"],
                    "y_values": modified_y_values,
                    "original_y_values": original["y_values"],
                }
            )

        return result


ml_service = MLService()
