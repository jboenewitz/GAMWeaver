"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PredictionInput(BaseModel):
    """Input schema for bike rental prediction."""
    temperature: float
    humidity: float
    windspeed: float
    time_of_day: int
    type_of_day: str
    weathersituation: str


class PredictionOutput(BaseModel):
    """Output schema for bike rental prediction."""
    predicted_count: float
    input_features: Dict[str, Any]


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    predictions: List[PredictionInput]


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[float]


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    rmse: float
    mae: float
    model_type: str


class ShapeFunctionData(BaseModel):
    """Shape function data for a single feature."""
    feature_name: str
    x_values: List[float]
    y_values: List[float]
    feature_type: str  # 'numeric' or 'categorical'


class DataSummary(BaseModel):
    """Summary statistics for the dataset."""
    total_records: int
    features: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    target_stats: Dict[str, float]


class FeatureImportance(BaseModel):
    """Feature importance data."""
    feature_name: str
    importance: float


class TrainModelRequest(BaseModel):
    """Request to train a model with specific parameters."""
    n_estimators: int = 100
    test_size: float = 0.2
    random_state: int = 42


class TrainModelResponse(BaseModel):
    """Response after training a model."""
    success: bool
    message: str
    metrics: Optional[ModelMetrics] = None


class EditedShapePoint(BaseModel):
    """A single edited point in a shape function."""
    x_value: Any  # Can be numeric or categorical
    y_value: float
    weight: float = 0.5  # Sureness weight (0.1 to 1.0, derived from 1-10 slider)
    message: str = ""  # Commit message for the edit


class EditedShapeFunction(BaseModel):
    """Edited shape function data for a single feature."""
    feature_name: str
    edited_points: List[EditedShapePoint]
    feature_type: str  # 'numeric' or 'categorical'


class EditedShapeFunctionsRequest(BaseModel):
    """Request with all edited shape functions."""
    edited_shape_functions: List[EditedShapeFunction]


class ComparisonMetrics(BaseModel):
    """Comparison metrics between original and interactive models."""
    original_rmse: float
    original_mae: float
    interactive_rmse: float
    interactive_mae: float


class PredictionComparisonResponse(BaseModel):
    """Response with original and interactive predictions."""
    original_predictions: List[float]
    interactive_predictions: List[float]
    actual_values: List[float]
    metrics: ComparisonMetrics


# ==================== User-related Models ====================

class UserLoginRequest(BaseModel):
    """Request to login or create a user."""
    name: str


class UserResponse(BaseModel):
    """Response with user information."""
    id: int
    name: str
    created_at: str
    is_new: bool = False


class UserListResponse(BaseModel):
    """Response with list of users."""
    users: List[UserResponse]


class UserEditsRequest(BaseModel):
    """Request to save user edits."""
    user_id: int
    edited_shape_functions: List[EditedShapeFunction]


class CombinedEditsResponse(BaseModel):
    """Response with combined edits from all users."""
    total_users_with_edits: int
    shape_functions: List[Dict[str, Any]]


class ResetDatabaseResponse(BaseModel):
    """Response after resetting the database."""
    success: bool
    message: str


class DeleteEditRequest(BaseModel):
    """Request to delete a specific edit."""
    edit_id: int
    deleted_by_user_id: int
    reason: str


class DeleteEditResponse(BaseModel):
    """Response after deleting an edit."""
    success: bool
    message: str


class EditDeletionNotification(BaseModel):
    """A notification about a deleted edit."""
    id: int
    feature_name: str
    x_value: str
    reason: str
    deleted_by: str
    created_at: str


class NotificationsResponse(BaseModel):
    """Response with list of notifications."""
    notifications: List[EditDeletionNotification]
