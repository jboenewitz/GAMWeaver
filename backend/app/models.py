"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Input schema for a generic single prediction."""

    input_features: Dict[str, Any]


class PredictionOutput(BaseModel):
    """Output schema for a generic single prediction."""

    predicted_value: float
    input_features: Dict[str, Any]


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    rmse: float
    mae: float
    model_type: str


class TrainModelRequest(BaseModel):
    """Request to train a model for a dataset."""

    user_id: int
    n_estimators: int = 100
    test_size: float = 0.2
    random_state: int = 42


class TrainModelResponse(BaseModel):
    """Response after training a model."""

    success: bool
    message: str
    metrics: Optional[ModelMetrics] = None
    model_version_id: Optional[int] = None


class EditedShapePoint(BaseModel):
    """A single edited point in a shape function."""

    x_value: Any
    y_value: float
    weight: float = 0.5
    message: str = ""


class EditedShapeFunction(BaseModel):
    """Edited shape function data for a single feature."""

    feature_name: str
    edited_points: List[EditedShapePoint]
    feature_type: str


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


class PredictionFieldSchema(BaseModel):
    """Schema for one prediction input field."""

    name: str
    label: str
    feature_type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    is_integer: Optional[bool] = None
    options: Optional[List[str]] = None


class DataSummary(BaseModel):
    """Summary statistics for the active dataset."""

    total_records: int
    features: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    target_label: str
    target_column: str
    target_stats: Dict[str, float]


class ModelStatus(BaseModel):
    """Resolved model status for the active dataset context."""

    dataset_id: Optional[int] = None
    model_version_id: Optional[int] = None
    dataset_name: Optional[str] = None
    version_number: Optional[int] = None
    data_loaded: bool
    is_trained: bool
    train_size: int = 0
    test_size: int = 0
    features: List[str] = []


class DatasetColumnInfo(BaseModel):
    """Detected column metadata for CSV inspection."""

    name: str
    is_numeric: bool
    non_null_count: int
    sample_values: List[str]


class DatasetInspectResponse(BaseModel):
    """Response for the dataset upload inspection step."""

    columns: List[DatasetColumnInfo]
    row_count: int


class DatasetInfo(BaseModel):
    """Persisted dataset metadata."""

    id: int
    display_name: str
    original_filename: str
    target_column: str
    created_at: str
    updated_at: str
    uploaded_by_user_id: Optional[int] = None
    latest_model_version_id: Optional[int] = None
    latest_model_version_number: Optional[int] = None


class ModelVersionInfo(BaseModel):
    """Persisted model version metadata."""

    id: int
    dataset_id: int
    version_number: int
    training_params: Dict[str, Any]
    train_size: int
    test_size: int
    schema_snapshot: Dict[str, Any]
    metrics: Dict[str, Any]
    created_by_user_id: Optional[int] = None
    created_at: str


class DatasetContextResponse(BaseModel):
    """Full active dataset context resolved for one user."""

    datasets: List[DatasetInfo]
    active_dataset: Optional[Dict[str, Any]] = None
    active_model_version: Optional[Dict[str, Any]] = None
    model_status: ModelStatus
    data_summary: Optional[DataSummary] = None
    prediction_fields: List[PredictionFieldSchema] = []
    capabilities: Dict[str, Any] = {}


class DatasetSelectionRequest(BaseModel):
    """Request to choose the active dataset for a user."""

    dataset_id: int


class UserLoginRequest(BaseModel):
    """Request to login a user."""

    username: str
    password: str


class UserResponse(BaseModel):
    """Response with user information."""

    id: int
    name: str
    created_at: str
    is_new: bool = False
    is_superadmin: bool = False
    access_token: Optional[str] = None


class UserRegisterRequest(BaseModel):
    """Request to register a new user via invite."""

    username: str
    password: str
    invite_token: str


class AdminCreateUserRequest(BaseModel):
    """Admin request to create a user."""

    username: str
    password: str


class InviteCreateResponse(BaseModel):
    """Response after creating an invite token."""

    token: str
    expires_at: Optional[str] = None


class UserListResponse(BaseModel):
    """Response with list of users."""

    users: List[UserResponse]


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
    model_version_id: Optional[int] = None


class NotificationsResponse(BaseModel):
    """Response with list of notifications."""

    notifications: List[EditDeletionNotification]


class UserPreferencesRequest(BaseModel):
    """Request to update user preferences."""

    preferences: Dict[str, Any]


class UserPreferencesResponse(BaseModel):
    """Response with user preferences."""

    preferences: Dict[str, Any]
