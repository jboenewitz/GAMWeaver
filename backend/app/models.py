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
    case_study_feature_columns: Optional[List[str]] = None


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
    x_axis_label: Optional[str] = None
    categorical_value_labels: Optional[Dict[str, str]] = None
    numeric_tick_labels: Optional[Dict[str, str]] = None
    value_labels: Optional[Dict[str, str]] = None


class FeatureChartSettingsResponse(BaseModel):
    """Response with effective chart-display settings for one feature."""
    feature_name: str
    display_title: str = ""
    x_axis_label: str = ""
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
    has_held_out_test_data: bool = False
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
    distribution: Optional[ShapeFunctionDistribution] = None


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


class ModelCompareHeldOutTestData(BaseModel):
    """Feature rows and targets reserved from training for artifact evaluation."""
    feature_rows: List[Dict[str, Any]] = Field(default_factory=list)
    target_values: List[float] = Field(default_factory=list)
    sample_ids: Optional[List[str]] = None


class ModelComparePreparedArtifact(BaseModel):
    """Canonical compare payload for one uploaded model artifact."""
    metadata: ModelCompareArtifactMetadata
    base_prediction_offset: float = 0.0
    feature_names: List[str] = Field(default_factory=list)
    selected_feature_columns: List[str]
    cat_features: List[str] = Field(default_factory=list)
    num_features: List[str] = Field(default_factory=list)
    missing_indicator_map: Dict[str, str] = Field(default_factory=dict)
    numeric_missing_placeholder_values: Dict[str, float] = Field(default_factory=dict)
    shape_functions: List[ModelCompareShapeFunction] = Field(default_factory=list)
    submissions_by_feature: List[ModelCompareFeatureSubmissions] = Field(
        default_factory=list
    )
    held_out_test_data: Optional[ModelCompareHeldOutTestData] = None
    training_feature_rows: Optional[List[Dict[str, Any]]] = None
    sample_id_namespace: Optional[str] = None
    training_sample_ids: Optional[List[str]] = None
    preprocessing_scope: Optional[str] = None


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


class ModelCompareSubmissionTrace(BaseModel):
    """One selected submission rendered independently on a feature curve."""
    submission_id: str
    user_name: str
    sureness: int = 0
    y_values: List[float]
    original_base_mae: Optional[float] = None
    original_base_mae_without_confidence: Optional[float] = None
    original_base_mae_with_confidence: Optional[float] = None


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
    right_alternate_confidence_y_values: List[float]
    left_individual_edit_traces: List[ModelCompareSubmissionTrace] = Field(
        default_factory=list
    )
    right_individual_edit_traces: List[ModelCompareSubmissionTrace] = Field(
        default_factory=list
    )
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
    limit_edited_mean_deviation: bool = False
    feature_names: Optional[List[str]] = None
    overall_mae_feature_names: Optional[List[str]] = None
    mean_deviation_feature_names: Optional[List[str]] = None
    mean_deviation_scaled_feature_names: Optional[List[str]] = None
    manual_feature_shifts: Optional[Dict[str, float]] = None
    intercept_adjustment: float = Field(default=0.0, allow_inf_nan=False)
    apply_training_intercept_correction: bool = False
    held_out_test_source: Optional[str] = None
    feature_metrics_on_original_test_set: bool = False


class ModelCompareAveragePredictionScores(BaseModel):
    """Average predictions on a shared curve grid or selected feature rows."""
    original: float
    reduced: float
    edited: float
    difference: float
    edited_minus_reduced: float


class ModelCompareTrainingMeanPredictionSummary(BaseModel):
    """Artifact-based means on the reduced model's preprocessed training rows."""
    training_row_count: int
    reduced_model_mean: float
    edited_before_correction_mean: float
    difference_before_correction: float
    edited_after_correction_mean: float
    original_full_data_intercept: float
    reduced_intercept: float
    edited_intercept_before_correction: float
    intercept_adjustment: float
    edited_intercept_after_correction: float
    suggested_matching_adjustment: float
    suggested_matching_intercept: float


class ModelCompareOverallShapeFunctionMaeMeasure(BaseModel):
    """One aggregate full-data shape-function MAE comparison."""
    baseline: float
    edited: float
    status: str


class ModelCompareExpertCompleteModelMae(BaseModel):
    """One expert's full selected model compared with the full-data curves."""
    user_name: str
    mae: float
    baseline_mae: float
    status: str
    feature_count: int
    point_count: int
    submission_count: int


class ModelCompareFeatureMaeChange(BaseModel):
    """One feature's MAE after all exported edits in a fixed confidence mode."""
    feature_name: str
    mae: float
    baseline_mae: float
    status: str
    point_count: int


class ModelCompareAllFeatureMaeChange(BaseModel):
    """Fixed per-feature comparisons in both confidence modes."""
    without_confidence: List[ModelCompareFeatureMaeChange] = Field(default_factory=list)
    with_confidence: List[ModelCompareFeatureMaeChange] = Field(default_factory=list)


class ModelCompareOverallShapeFunctionMae(BaseModel):
    """Pooled and equal-feature summaries of shape-function MAE."""
    feature_count: int
    point_count: int
    pooled: ModelCompareOverallShapeFunctionMaeMeasure
    mean_per_feature: ModelCompareOverallShapeFunctionMaeMeasure
    expert_complete_model_mae: List[ModelCompareExpertCompleteModelMae] = Field(
        default_factory=list
    )


class ModelCompareHeldOutTestMetrics(BaseModel):
    """Predictive metrics evaluated on one fixed held-out test set."""
    mae: float
    rmse: float
    std_dev: float
    mean_error: float
    r2: Optional[float]


class ModelCompareHeldOutTestPerformance(BaseModel):
    """Original and edited artifact accuracy on a shared selected dataset."""
    test_size: int
    test_data_source: str
    baseline: ModelCompareHeldOutTestMetrics
    reduced: ModelCompareHeldOutTestMetrics
    edited: ModelCompareHeldOutTestMetrics
    edited_before_intercept_correction: Optional[ModelCompareHeldOutTestMetrics] = None
    actual_values: List[float] = Field(default_factory=list)
    baseline_predictions: List[float] = Field(default_factory=list)
    reduced_predictions: List[float] = Field(default_factory=list)
    edited_predictions: List[float] = Field(default_factory=list)
    edited_before_intercept_predictions: Optional[List[float]] = None
    edit_effect: Dict[str, Any] = Field(default_factory=dict)
    paired_bootstrap: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    intercept_correction: Dict[str, Any] = Field(default_factory=dict)


class ModelCompareAverageFeatureContribution(BaseModel):
    """One additive term behind an average-prediction score."""
    feature_name: str
    original: float
    reduced: float
    edited: float
    difference: float
    is_base_offset: bool = False
    is_internal_feature: bool = False


class ModelCompareMeanDeviationConstraint(BaseModel):
    """Preview-only scaling applied to keep edited-average drift bounded."""
    enabled: bool
    applied: bool
    scale_factor: float
    limit: float
    original_average: float
    unconstrained_edited_average: float
    constrained_edited_average: float
    satisfied: bool
    feature_scale_factors: Dict[str, float] = Field(default_factory=dict)


class ModelComparePreviewResponse(BaseModel):
    """Response with recalculated preview traces for compared features."""
    use_confidence: bool = True
    intercept_adjustment: float = 0.0
    original_edited_base_prediction_offset: float = 0.0
    adjusted_edited_base_prediction_offset: float = 0.0
    feature_previews: List[ModelCompareFeaturePreview] = Field(default_factory=list)
    all_feature_mae_change: ModelCompareAllFeatureMaeChange
    overall_shape_function_mae: ModelCompareOverallShapeFunctionMae
    held_out_test_performance: Optional[ModelCompareHeldOutTestPerformance] = None
    main_training_performance_available: bool = False
    average_prediction_scores: ModelCompareAveragePredictionScores
    training_mean_prediction_summary: Optional[ModelCompareTrainingMeanPredictionSummary] = None
    test_set_average_prediction_scores: Optional[ModelCompareAveragePredictionScores] = None
    main_dataset_average_prediction_scores: Optional[ModelCompareAveragePredictionScores] = None
    main_dataset_training_row_count: int = 0
    main_dataset_average_prediction_shift: Optional[float] = None
    average_prediction_feature_contributions: Dict[
        str, Optional[List[ModelCompareAverageFeatureContribution]]
    ] = Field(default_factory=dict)
    mean_deviation_constraint: ModelCompareMeanDeviationConstraint


class ModelComparePrepareResponse(BaseModel):
    """Response after validating two uploaded model artifacts for comparison."""
    left_artifact: ModelComparePreparedArtifact
    right_artifact: ModelComparePreparedArtifact
    shared_features: List[str] = Field(default_factory=list)
    preview: ModelComparePreviewResponse
