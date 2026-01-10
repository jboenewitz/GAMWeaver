"""Machine Learning service for IGANN model training and prediction."""

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from igann import IGANN
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
from pathlib import Path

from .data_processing import prepare_training_data, preprocess_data, get_preprocessor

# Determine the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_FILE = PROJECT_ROOT / "bike.csv"


class MLService:
    """Service for managing IGANN models and predictions."""
    
    def __init__(self):
        self.model: Optional[IGANN] = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.is_trained = False
        self.feature_names = []
        self.cat_features = ["Weathersituation", "Time of Day", "Type of Day"]
        self.num_features = ["Temperature", "Humidity", "Windspeed"]
        # Store original shape functions for interactive editing
        self.original_shape_functions: Dict[str, Dict[str, Any]] = {}
        self.shape_function_offsets: Dict[str, Dict[Any, float]] = {}
    
    def load_data(self, csv_path: str = None) -> Dict[str, Any]:
        """Load and prepare the dataset."""
        if csv_path is None:
            csv_path = DATA_FILE
        
        # Ensure the file exists
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Dataset not found at: {csv_path}. Please ensure bike.csv is in the project root.")
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.preprocessor, self.df = \
            prepare_training_data(csv_path)
        
        self.feature_names = list(self.X_train.columns)
        
        return {
            "total_records": len(self.df),
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "features": self.feature_names,
        }
    
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
        
        # Calculate metrics
        metrics = self.evaluate_model()
        return metrics
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "model_type": "IGANN"
        }
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Make a single prediction."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Create DataFrame with feature names
        input_df = pd.DataFrame([{
            "Temperature": features["temperature"],
            "Humidity": features["humidity"],
            "Windspeed": features["windspeed"],
            "Time of Day": str(features["time_of_day"]),
            "Type of Day": features["type_of_day"],
            "Weathersituation": features["weathersituation"],
        }])
        
        # Ensure correct types
        input_df = input_df.astype({
            "Time of Day": "object",
            "Type of Day": "object",
            "Weathersituation": "object",
        })
        
        prediction = self.model.predict(input_df)
        return float(prediction[0])
    
    def batch_predict(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Make batch predictions."""
        return [self.predict(f) for f in features_list]
    
    def get_shape_functions(self) -> List[Dict[str, Any]]:
        """Get shape function data for all features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        shape_functions = []
        
        # Get shape functions from the model
        for i, feature_name in enumerate(self.feature_names):
            try:
                # Get the shape function data from the model
                shape_data = self._extract_shape_function(feature_name, i)
                shape_functions.append(shape_data)
                # Store for later use in interactive predictions
                self.original_shape_functions[feature_name] = shape_data
            except Exception as e:
                print(f"Error extracting shape function for {feature_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Reset offsets when getting fresh shape functions
        self.shape_function_offsets = {}
        
        return shape_functions
    
    def _extract_shape_function(self, feature_name: str, feature_idx: int) -> Dict[str, Any]:
        """Extract shape function data for a single feature."""
        is_categorical = feature_name in self.cat_features
        
        # Get baseline values for all features
        baseline = {}
        for num_feat in self.num_features:
            baseline[num_feat] = float(self.X_train[num_feat].mean())
        for cat_feat in self.cat_features:
            baseline[cat_feat] = str(self.X_train[cat_feat].mode().iloc[0])
        
        if is_categorical:
            # For categorical features, get unique values
            unique_values = [str(v) for v in self.X_train[feature_name].unique().tolist()]
            
            shape_values = []
            for val in unique_values:
                # Create a sample with this category value
                sample_data = baseline.copy()
                sample_data[feature_name] = val
                
                # Create DataFrame with correct column order
                sample = pd.DataFrame([sample_data])[self.feature_names]
                
                # Ensure correct types
                for cat_feat in self.cat_features:
                    sample[cat_feat] = sample[cat_feat].astype(str)
                for num_feat in self.num_features:
                    sample[num_feat] = sample[num_feat].astype(float)
                
                pred = self.model.predict(sample)
                pred_val = float(pred[0]) if hasattr(pred, '__iter__') else float(pred)
                shape_values.append(pred_val)
            
            # Normalize to show relative effect
            mean_val = np.mean(shape_values)
            shape_values = [v - mean_val for v in shape_values]
            
            return {
                "feature_name": feature_name,
                "x_values": unique_values,
                "y_values": shape_values,
                "feature_type": "categorical"
            }
        else:
            # For numeric features, create a range of values
            min_val = float(self.X_train[feature_name].min())
            max_val = float(self.X_train[feature_name].max())
            x_range = np.linspace(min_val, max_val, 30)  # Reduced for performance
            
            shape_values = []
            for x_val in x_range:
                # Create a sample with this numeric value
                sample_data = baseline.copy()
                sample_data[feature_name] = float(x_val)
                
                # Create DataFrame with correct column order
                sample = pd.DataFrame([sample_data])[self.feature_names]
                
                # Ensure correct types
                for cat_feat in self.cat_features:
                    sample[cat_feat] = sample[cat_feat].astype(str)
                for num_feat in self.num_features:
                    sample[num_feat] = sample[num_feat].astype(float)
                
                pred = self.model.predict(sample)
                pred_val = float(pred[0]) if hasattr(pred, '__iter__') else float(pred)
                shape_values.append(pred_val)
            
            # Normalize
            mean_val = np.mean(shape_values)
            shape_values = [v - mean_val for v in shape_values]
            
            return {
                "feature_name": feature_name,
                "x_values": x_range.tolist(),
                "y_values": shape_values,
                "feature_type": "numeric"
            }
    
    def get_predictions_vs_actual(self) -> Dict[str, List[float]]:
        """Get predictions vs actual values for visualization."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(self.X_test)
        y_actual = self.y_test.values.flatten().tolist()
        
        return {
            "predicted": [float(p) for p in y_pred],
            "actual": y_actual
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        if self.df is None:
            self.load_data()
        
        return {
            "total_records": len(self.df),
            "features": self.feature_names,
            "numeric_features": self.num_features,
            "categorical_features": self.cat_features,
            "target_stats": {
                "mean": float(self.df["cnt"].mean()),
                "std": float(self.df["cnt"].std()),
                "min": float(self.df["cnt"].min()),
                "max": float(self.df["cnt"].max()),
            }
        }
    
    def get_feature_distributions(self) -> Dict[str, Any]:
        """Get feature distributions for visualization."""
        if self.X_train is None:
            self.load_data()
        
        distributions = {}
        
        for feature in self.num_features:
            values = self.X_train[feature].values.tolist()
            distributions[feature] = {
                "type": "numeric",
                "values": values[:1000],  # Limit for performance
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        for feature in self.cat_features:
            value_counts = self.X_train[feature].value_counts().to_dict()
            distributions[feature] = {
                "type": "categorical",
                "counts": {str(k): int(v) for k, v in value_counts.items()}
            }
        
        return distributions
    
    def get_hourly_pattern(self) -> Dict[str, Any]:
        """Get hourly bike rental pattern."""
        if self.df is None:
            self.load_data()
        
        hourly_avg = self.df.groupby("hr")["cnt"].mean()
        
        return {
            "hours": hourly_avg.index.tolist(),
            "avg_rentals": hourly_avg.values.tolist()
        }
    
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
            
            # Create offset mapping for this feature
            self.shape_function_offsets[feature_name] = {}
            
            for edited_point in edited_points:
                x_val = edited_point["x_value"]
                new_y = edited_point["y_value"]
                
                # Find the closest original x value and its y value
                if feature_type == "categorical":
                    # For categorical, match by string value
                    x_str = str(x_val)
                    if x_str in original_x:
                        idx = original_x.index(x_str)
                        original_y_val = original_y[idx]
                        offset = new_y - original_y_val
                        self.shape_function_offsets[feature_name][x_str] = offset
                else:
                    # For numeric, find closest x value
                    x_float = float(x_val)
                    closest_idx = np.argmin(np.abs(np.array(original_x) - x_float))
                    original_y_val = original_y[closest_idx]
                    offset = new_y - original_y_val
                    self.shape_function_offsets[feature_name][closest_idx] = offset
    
    def _get_offset_for_value(self, feature_name: str, value: Any) -> float:
        """Get the offset to apply for a given feature value."""
        if feature_name not in self.shape_function_offsets:
            return 0.0
        
        offsets = self.shape_function_offsets[feature_name]
        
        if feature_name in self.cat_features:
            # Categorical: direct lookup
            return offsets.get(str(value), 0.0)
        else:
            # Numeric: interpolate between edited points
            if not offsets:
                return 0.0
            
            original = self.original_shape_functions.get(feature_name, {})
            original_x = original.get("x_values", [])
            
            if not original_x:
                return 0.0
            
            # Find where this value falls in the original x range
            value_float = float(value)
            
            # If we have offsets, interpolate
            indices = sorted(offsets.keys())
            offset_values = [offsets[i] for i in indices]
            x_positions = [original_x[i] for i in indices]
            
            if len(indices) == 1:
                return offset_values[0]
            
            # Linear interpolation
            if value_float <= x_positions[0]:
                return offset_values[0]
            if value_float >= x_positions[-1]:
                return offset_values[-1]
            
            # Find bracketing points
            for i in range(len(x_positions) - 1):
                if x_positions[i] <= value_float <= x_positions[i + 1]:
                    # Linear interpolation
                    t = (value_float - x_positions[i]) / (x_positions[i + 1] - x_positions[i])
                    return offset_values[i] + t * (offset_values[i + 1] - offset_values[i])
            
            return 0.0
    
    def predict_interactive(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with interactive shape function modifications."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Get base predictions
        base_predictions = self.model.predict(X)
        
        # Apply offsets based on feature values
        offsets = np.zeros(len(X))
        
        for feature_name in self.shape_function_offsets:
            if feature_name in X.columns:
                for i, value in enumerate(X[feature_name].values):
                    offsets[i] += self._get_offset_for_value(feature_name, value)
        
        return base_predictions + offsets
    
    def get_predictions_comparison(self) -> Dict[str, Any]:
        """Get comparison between original and interactive predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Original predictions
        y_pred_original = self.model.predict(self.X_test)
        
        # Interactive predictions (with offsets)
        y_pred_interactive = self.predict_interactive(self.X_test)
        
        y_actual = self.y_test.values.flatten()
        
        # Calculate metrics
        original_rmse = root_mean_squared_error(y_actual, y_pred_original)
        original_mae = mean_absolute_error(y_actual, y_pred_original)
        
        interactive_rmse = root_mean_squared_error(y_actual, y_pred_interactive)
        interactive_mae = mean_absolute_error(y_actual, y_pred_interactive)
        
        return {
            "original_predictions": [float(p) for p in y_pred_original],
            "interactive_predictions": [float(p) for p in y_pred_interactive],
            "actual_values": y_actual.tolist(),
            "metrics": {
                "original_rmse": float(original_rmse),
                "original_mae": float(original_mae),
                "interactive_rmse": float(interactive_rmse),
                "interactive_mae": float(interactive_mae),
            }
        }


# Global service instance
ml_service = MLService()
