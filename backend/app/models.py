"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ShapeFunctionDistributionBin(BaseModel):
    """A single numeric distribution bin."""
    x0: float
    x1: float
    count: int
    center: float


class ShapeFunctionDistributionCount(BaseModel):
    """A single categorical distribution count."""
    x_value: Any
    label: str
    count: int


class ShapeFunctionDistribution(BaseModel):
    """Distribution metadata shown below an interactive shape function."""
    chart_type: str  # 'numeric' or 'categorical'
    total_count: int
    bin_count: int
    bins: List[ShapeFunctionDistributionBin] = Field(default_factory=list)
    counts: List[ShapeFunctionDistributionCount] = Field(default_factory=list)
    missing_count: int = 0
    missing_label: Optional[str] = None


class ShapeFunctionMissingBucket(BaseModel):
    """Dedicated missing-value bar shown alongside a numeric shape function."""
    label: str
    count: int
    y_value: float


class PredictionInput(BaseModel):
    """Dynamic input schema for single prediction."""
    features: Dict[str, Any]


class PredictionOutput(BaseModel):
    """Output schema for bike rental prediction."""
    predicted_count: float
    input_features: Dict[str, Any]


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    predictions: List[Dict[str, Any]]


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
    x_values: List[Any]
    y_values: List[float]
    feature_type: str  # 'numeric' or 'categorical'
    x_tick_labels: Optional[List[str]] = None
    chart_config: Optional[Dict[str, Any]] = None
    distribution: Optional[ShapeFunctionDistribution] = None
    missing_bucket: Optional[ShapeFunctionMissingBucket] = None


class ShapeFunctionsResponse(BaseModel):
    """Response with primary and comparison shape function data."""
    shape_functions: List[ShapeFunctionData]
    comparison_shape_functions: List[ShapeFunctionData]


class DataSummary(BaseModel):
    """Summary statistics for the dataset."""
    total_records: int
    features: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    target_column: str
    target_stats: Dict[str, float]


class DataLoadRequest(BaseModel):
    """Request to load a selected dataset and target column."""
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None


class DatasetUploadResponse(BaseModel):
    """Response after uploading/inspecting CSV dataset."""
    dataset_id: str
    original_filename: str
    columns: List[str]
    default_target_column: str


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


class ModelImportResponse(BaseModel):
    """Response after importing a model artifact."""
    success: bool
    message: str
    model_source: str
    imported_artifact_version: Optional[str] = None
    imported_shape_function_edits: bool = False
    imported_edit_user_count: int = 0
    imported_edit_submission_count: int = 0


class ModelStatusResponse(BaseModel):
    """Response describing the currently active model/runtime state."""
    is_trained: bool
    data_loaded: bool
    features: List[str]
    feature_schema: List[Dict[str, Any]]
    target_column: Optional[str] = None
    selected_feature_columns: List[str]
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    train_size: int = 0
    test_size: int = 0
    model_source: str = "trained"
    analytics_available: bool = False
    shape_functions_available: bool = False
    imported_artifact_version: Optional[str] = None
    comparison_available: bool = False
    comparison_data_loaded: bool = False
    comparison_is_trained: bool = False
    comparison_dataset_name: Optional[str] = None
    comparison_train_size: int = 0
    primary_n_estimators: Optional[int] = None
    show_missing_bars: bool = False
    show_competence_levels: bool = False


class ComparisonDataLoadRequest(BaseModel):
    """Request to load a selected comparison dataset."""
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None


class ComparisonTrainResponse(BaseModel):
    """Response after training the comparison model."""
    success: bool
    message: str
    inherited_n_estimators: int


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
    """Request to login a user."""
    username: str
    password: str


class UserResponse(BaseModel):
    """Response with user information."""
    id: int
    name: str
    profession: Optional[str] = None
    created_at: str
    is_new: bool = False
    is_superadmin: bool = False
    access_token: Optional[str] = None


class UserRegisterRequest(BaseModel):
    """Request to register a new user via invite."""
    username: str
    password: str
    invite_token: str
    profession: str


class AdminCreateUserRequest(BaseModel):
    """Admin request to create a user."""
    username: str
    password: str
    profession: str


class InviteCreateResponse(BaseModel):
    """Response after creating an invite token."""
    token: str
    expires_at: Optional[str] = None


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


class DeleteSubmissionRequest(BaseModel):
    """Request to delete a full curve-edit submission."""
    submission_id: str
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
    submission_id: Optional[str] = None
    point_count: Optional[int] = None
    x_summary: Optional[str] = None
    reason: str
    deleted_by: str
    created_at: str


class NotificationsResponse(BaseModel):
    """Response with list of notifications."""
    notifications: List[EditDeletionNotification]


class UserPreferencesRequest(BaseModel):
    """Request to update user preferences."""
    preferences: Dict[str, Any]


class UserPreferencesResponse(BaseModel):
    """Response with user preferences."""
    preferences: Dict[str, Any]


class FeatureChartSettingsRequest(BaseModel):
    """Superadmin chart-display settings for one feature."""
    treat_as_categorical: bool = False
    treat_as_numeric: bool = False
    display_title: Optional[str] = None
    categorical_value_labels: Optional[Dict[str, str]] = None
    numeric_tick_labels: Optional[Dict[str, str]] = None
    value_labels: Optional[Dict[str, str]] = None


class FeatureChartSettingsResponse(BaseModel):
    """Response with effective chart-display settings for one feature."""
    feature_name: str
    display_title: str = ""
    base_feature_type: str
    chart_feature_type: str
    is_numeric: bool
    is_categorical: bool
    can_be_categorical: bool
    can_be_numeric: bool
    treat_as_categorical: bool
    treat_as_numeric: bool
    categorical_value_labels: Dict[str, str]
    numeric_tick_labels: Dict[str, str]
    value_labels: Dict[str, str] = {}
    available_categorical_values: List[str] = []
    available_numeric_values: List[str] = []


class ChartDisplaySettingsRequest(BaseModel):
    """Global superadmin chart-display settings."""
    show_missing_bars: bool = False
    show_competence_levels: bool = False


class ChartDisplaySettingsResponse(BaseModel):
    """Response with global chart-display settings."""
    show_missing_bars: bool
    show_competence_levels: bool


class ModelCompareArtifactMetadata(BaseModel):
    """Summary metadata for one uploaded model artifact."""
    filename: str
    exported_at: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_id: Optional[str] = None
    target_column: str
    selected_feature_count: int
    has_edit_export: bool = False
    edit_user_count: int = 0
    edit_submission_count: int = 0
    model_source: Optional[str] = None
    artifact_version: Optional[str] = None


class ModelCompareShapeFunction(BaseModel):
    """Canonical base shape function extracted from an artifact."""
    feature_name: str
    feature_type: str
    x_values: List[Any]
    y_values: List[float]
    x_tick_labels: Optional[List[str]] = None
    chart_config: Optional[Dict[str, Any]] = None


class ModelCompareEditedPoint(BaseModel):
    """One stored edit point included in a compare submission."""
    x_value: Any
    y_value: float
    weight: float = 0.5
    message: str = ""
    display_x_value: Optional[Any] = None


class ModelCompareSubmission(BaseModel):
    """One imported user edit submission available for preview selection."""
    submission_id: str
    feature_name: str
    feature_type: str
    user_name: str
    profession: Optional[str] = None
    message: str = ""
    sureness: int = 0
    point_count: int = 0
    x_summary: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    edited_points: List[ModelCompareEditedPoint] = Field(default_factory=list)


class ModelCompareFeatureSubmissions(BaseModel):
    """Submission tree grouped by feature for the compare page."""
    feature_name: str
    feature_type: str
    submissions: List[ModelCompareSubmission] = Field(default_factory=list)


class ModelComparePreparedArtifact(BaseModel):
    """Canonical compare payload for one uploaded model artifact."""
    metadata: ModelCompareArtifactMetadata
    selected_feature_columns: List[str]
    cat_features: List[str] = Field(default_factory=list)
    num_features: List[str] = Field(default_factory=list)
    shape_functions: List[ModelCompareShapeFunction] = Field(default_factory=list)
    submissions_by_feature: List[ModelCompareFeatureSubmissions] = Field(
        default_factory=list
    )


class ModelCompareFeatureCurveMetrics(BaseModel):
    """Per-feature curve comparison metrics for the edited artifact preview."""
    edited_vs_edited_base_mae: float
    edited_vs_original_base_mae: float
    baseline_edited_vs_edited_base_mae: float
    baseline_edited_vs_original_base_mae: float
    edited_vs_edited_base_status: str
    edited_vs_original_base_status: str
    edited_vs_edited_base_rmse: float
    edited_vs_original_base_rmse: float
    baseline_edited_vs_edited_base_rmse: float
    baseline_edited_vs_original_base_rmse: float
    edited_vs_edited_base_rmse_status: str
    edited_vs_original_base_rmse_status: str
    edited_vs_edited_base_r2: float
    edited_vs_original_base_r2: float
    baseline_edited_vs_edited_base_r2: float
    baseline_edited_vs_original_base_r2: float
    edited_vs_edited_base_r2_status: str
    edited_vs_original_base_r2_status: str


class ModelCompareFeaturePreview(BaseModel):
    """Combined left/right preview for one feature."""
    feature_name: str
    feature_type: str
    left_x_values: List[Any]
    left_x_tick_labels: Optional[List[str]] = None
    right_x_values: List[Any]
    right_x_tick_labels: Optional[List[str]] = None
    left_base_y_values: List[float]
    left_effective_y_values: List[float]
    right_base_y_values: List[float]
    right_effective_y_values: List[float]
    left_chart_config: Optional[Dict[str, Any]] = None
    right_chart_config: Optional[Dict[str, Any]] = None
    comparison_metrics: Optional[ModelCompareFeatureCurveMetrics] = None


class ModelComparePreviewRequest(BaseModel):
    """Request to recompute a compare preview from canonical artifacts."""
    left_artifact: ModelComparePreparedArtifact
    right_artifact: ModelComparePreparedArtifact
    left_selected_submission_ids: List[str] = Field(default_factory=list)
    right_selected_submission_ids: List[str] = Field(default_factory=list)
    use_confidence: bool = True
    feature_names: Optional[List[str]] = None


class ModelComparePreviewResponse(BaseModel):
    """Response with recalculated preview traces for compared features."""
    use_confidence: bool = True
    feature_previews: List[ModelCompareFeaturePreview] = Field(default_factory=list)


class ModelComparePrepareResponse(BaseModel):
    """Response after validating two uploaded model artifacts for comparison."""
    left_artifact: ModelComparePreparedArtifact
    right_artifact: ModelComparePreparedArtifact
    shared_features: List[str] = Field(default_factory=list)
    preview: ModelComparePreviewResponse
