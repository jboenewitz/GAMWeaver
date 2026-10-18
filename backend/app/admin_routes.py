"""Superadmin menu API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile

from .api_common import _normalize_required_profession, _read_uploaded_json_artifact, _require_destructive_access, _require_superadmin
from .config import settings
from .db_service import db_service
from .ml_service import ml_service
from .models import (
    AdminCreateUserRequest,
    ChartDisplaySettingsRequest,
    ChartDisplaySettingsResponse,
    ComparisonDataLoadRequest,
    ComparisonTrainResponse,
    DataLoadRequest,
    DatasetUploadResponse,
    DeleteEditRequest,
    DeleteEditResponse,
    DeleteSubmissionRequest,
    FeatureChartSettingsRequest,
    FeatureChartSettingsResponse,
    InviteCreateResponse,
    ModelImportResponse,
    ModelMetrics,
    ResetDatabaseResponse,
    TrainModelRequest,
    TrainModelResponse,
    UserListResponse,
    UserResponse,
)
from .preprocessing_sensitivity import compare_models as compare_preprocessing_sensitivity

router = APIRouter()


@router.post("/api/model/preprocessing-sensitivity")
async def preprocessing_sensitivity(
    request: Request,
    full_csv: UploadFile = File(...),
    full_model: UploadFile = File(...),
    reduced_csv: UploadFile = File(...),
    reduced_model: UploadFile = File(...),
):
    """Retrain unedited IQB models with medians learned on training rows only."""
    _require_superadmin(request)
    for csv_file in (full_csv, reduced_csv):
        if not (csv_file.filename or "").lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Upload CSV datasets")
    full_artifact = await _read_uploaded_json_artifact(full_model)
    reduced_artifact = await _read_uploaded_json_artifact(reduced_model)
    for csv_file, artifact in (
        (full_csv, full_artifact),
        (reduced_csv, reduced_artifact),
    ):
        if isinstance(artifact, dict) and artifact.get("dataset_name"):
            if Path(csv_file.filename or "").name != Path(artifact["dataset_name"]).name:
                raise HTTPException(
                    status_code=400,
                    detail="Each CSV filename must match its original model export",
                )
    try:
        return compare_preprocessing_sensitivity(
            await full_csv.read(), full_artifact,
            await reduced_csv.read(), reduced_artifact,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/data/upload", response_model=DatasetUploadResponse)
async def upload_data(
    request: Request,
    file: UploadFile = File(...),
    use_case_study_feature_names: bool = Form(False),
):
    """Upload CSV and return columns for target selection (superadmin only)."""
    try:
        _require_superadmin(request)
        file_name = file.filename or "dataset.csv"
        file_bytes = await file.read()
        preview = ml_service.inspect_uploaded_dataset(
            file_name,
            file_bytes,
            use_case_study_feature_names=use_case_study_feature_names,
        )
        return DatasetUploadResponse(**preview)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/data/load")
async def load_data(http_request: Request, request: Optional[DataLoadRequest] = None):
    """Load and prepare the selected dataset (superadmin only)."""
    try:
        _require_superadmin(http_request)
        if request is None:
            request = DataLoadRequest()

        result = ml_service.load_data(
            dataset_id=request.dataset_id,
            dataset_name=request.dataset_name,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
        )
        if result.get("dataset_changed"):
            db_service.clear_all_shape_edits()
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/comparison/upload", response_model=DatasetUploadResponse)
async def upload_comparison_data(request: Request, file: UploadFile = File(...)):
    """Upload CSV and return columns for comparison target selection (superadmin only)."""
    try:
        _require_superadmin(request)
        file_name = file.filename or "dataset.csv"
        file_bytes = await file.read()
        preview = ml_service.inspect_uploaded_dataset(file_name, file_bytes)
        return DatasetUploadResponse(**preview)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/comparison/load")
async def load_comparison_data(
    http_request: Request,
    request: Optional[ComparisonDataLoadRequest] = None,
):
    """Load and prepare a comparison dataset without replacing the active dataset."""
    try:
        _require_superadmin(http_request)
        if request is None:
            request = ComparisonDataLoadRequest()
        result = ml_service.load_comparison_data(
            dataset_id=request.dataset_id,
            dataset_name=request.dataset_name,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
        )
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model/train", response_model=TrainModelResponse)
async def train_model(http_request: Request, request: Optional[TrainModelRequest] = None):
    """Train the IGANN model (superadmin only)."""
    try:
        _require_superadmin(http_request)
        if request is None:
            request = TrainModelRequest()
        
        # Load data if not already loaded
        if ml_service.X_train is None:
            ml_service.load_data()
        
        # Train the model
        metrics = ml_service.train_model(n_estimators=request.n_estimators)
        
        return TrainModelResponse(
            success=True,
            message="Model trained successfully",
            metrics=ModelMetrics(**metrics)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/comparison/train", response_model=ComparisonTrainResponse)
async def train_comparison_model(http_request: Request):
    """Train the comparison IGANN model using the primary model's estimator count."""
    try:
        _require_superadmin(http_request)
        inherited_n_estimators = ml_service.train_comparison_model()
        return ComparisonTrainResponse(
            success=True,
            message="Comparison model trained successfully",
            inherited_n_estimators=inherited_n_estimators,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/model/export")
async def export_model(http_request: Request, include_edits: bool = False):
    """Export the active model artifact as JSON (superadmin only)."""
    try:
        _require_superadmin(http_request)
        artifact = ml_service.export_model_artifact(
            include_shape_function_edits=include_edits
        )
        if include_edits:
            artifact["shape_function_edits_export"] = (
                db_service.export_shape_function_edits_artifact(
                    shape_functions=artifact["shape_functions"],
                )
            )
        timestamp = artifact.get("exported_at", "").replace(":", "-").replace(".", "-")
        suffix = "-with-edits" if include_edits else ""
        file_name = f"igann-model{suffix}-{timestamp}.json"
        return Response(
            content=json.dumps(artifact, ensure_ascii=True, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{file_name}"',
            },
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model/import", response_model=ModelImportResponse)
async def import_model(http_request: Request, file: UploadFile = File(...)):
    """Import a model artifact from JSON (superadmin only)."""
    try:
        _require_superadmin(http_request)
        file_name = (file.filename or "").lower()
        if file_name and not file_name.endswith(".json"):
            raise HTTPException(status_code=400, detail="Only JSON model artifacts are supported")
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            artifact = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON artifact: {exc}") from exc
        result = ml_service.import_model_artifact(artifact)
        imported_edits_payload = result.pop("_shape_function_edits_export", {})
        if result.get("imported_shape_function_edits"):
            import_counts = db_service.import_shape_function_edits_artifact(
                imported_edits_payload
            )
            result.update(import_counts)
        else:
            db_service.clear_all_shape_edits()
        return ModelImportResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/model/feature-chart-settings/{feature_name}",
    response_model=FeatureChartSettingsResponse,
)
async def get_feature_chart_settings(feature_name: str):
    """Get effective chart display settings for one feature."""
    try:
        settings_payload = ml_service.get_feature_chart_setting(feature_name)
        return FeatureChartSettingsResponse(**settings_payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/api/model/feature-chart-settings/{feature_name}",
    response_model=FeatureChartSettingsResponse,
)
async def update_feature_chart_settings(
    feature_name: str,
    request: FeatureChartSettingsRequest,
    http_request: Request,
):
    """Update superadmin chart display settings for one feature."""
    try:
        _require_superadmin(http_request)
        settings_payload = ml_service.update_feature_chart_setting(
            feature_name=feature_name,
            treat_as_categorical=request.treat_as_categorical,
            treat_as_numeric=request.treat_as_numeric,
            display_title=request.display_title,
            x_axis_label=request.x_axis_label,
            categorical_value_labels=request.categorical_value_labels,
            numeric_tick_labels=request.numeric_tick_labels,
            value_labels=request.value_labels,
        )
        return FeatureChartSettingsResponse(**settings_payload)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/model/chart-display-settings",
    response_model=ChartDisplaySettingsResponse,
)
async def get_chart_display_settings():
    """Get global chart display settings."""
    try:
        payload = ml_service.get_chart_display_settings()
        return ChartDisplaySettingsResponse(**payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/api/model/chart-display-settings",
    response_model=ChartDisplaySettingsResponse,
)
async def update_chart_display_settings(
    request: ChartDisplaySettingsRequest,
    http_request: Request,
):
    """Update global superadmin chart display settings."""
    try:
        _require_superadmin(http_request)
        payload = ml_service.update_chart_display_settings(
            show_missing_bars=request.show_missing_bars,
            show_competence_levels=request.show_competence_levels,
            show_prediction_comparison_superadmin=(
                request.show_prediction_comparison_superadmin
            ),
            show_prediction_comparison_users=request.show_prediction_comparison_users,
        )
        return ChartDisplaySettingsResponse(**payload)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/admin/users", response_model=UserListResponse)
async def get_all_users_admin(request: Request):
    """Get all users (superadmin only)."""
    try:
        _require_superadmin(request)
        users = db_service.get_all_users()
        return UserListResponse(users=users)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/admin/users", response_model=UserResponse)
async def create_user_admin(request_body: AdminCreateUserRequest, request: Request):
    """Create a user (superadmin only)."""
    try:
        _require_superadmin(request)
        username = request_body.username.strip()
        if not username or not request_body.password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        profession = _normalize_required_profession(request_body.profession)
        user = db_service.create_user_with_password(
            username,
            request_body.password,
            profession=profession,
        )
        return UserResponse(**user)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/admin/invites", response_model=InviteCreateResponse)
async def create_invite(request: Request):
    """Create a registration invite (superadmin only)."""
    try:
        _require_superadmin(request)
        admin_user = db_service.get_user_by_name(settings.superadmin_username)
        admin_id = admin_user["id"] if admin_user else None
        invite = db_service.create_invite_token(admin_id)
        return InviteCreateResponse(**invite)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/edits/delete", response_model=DeleteEditResponse)
async def delete_edit(request: DeleteEditRequest, http_request: Request):
    """Delete a specific edit and optionally notify the edit owner."""
    try:
        _require_destructive_access(http_request)
        if not request.reason or not request.reason.strip():
            raise HTTPException(status_code=400, detail="Reason is required")
        
        success = db_service.delete_edit(
            edit_id=request.edit_id,
            deleted_by_user_id=request.deleted_by_user_id,
            reason=request.reason.strip()
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Edit not found")
        
        return DeleteEditResponse(success=True, message="Edit deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/edits/delete-submission", response_model=DeleteEditResponse)
async def delete_submission(request: DeleteSubmissionRequest, http_request: Request):
    """Delete a full curve-edit submission and optionally notify the owner."""
    try:
        _require_destructive_access(http_request)
        if not request.reason or not request.reason.strip():
            raise HTTPException(status_code=400, detail="Reason is required")

        success = db_service.delete_submission(
            submission_id=request.submission_id,
            deleted_by_user_id=request.deleted_by_user_id,
            reason=request.reason.strip(),
        )

        if not success:
            raise HTTPException(status_code=404, detail="Submission not found")

        return DeleteEditResponse(
            success=True,
            message="Edit submission deleted successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/database/reset", response_model=ResetDatabaseResponse)
async def reset_database(http_request: Request):
    """Reset DB data and persisted uploaded-dataset artifacts."""
    try:
        _require_destructive_access(http_request)
        db_service.reset_all_data()
        cleanup_result = ml_service.reset_all_data()
        return ResetDatabaseResponse(
            success=True,
            message=(
                "Database reset successfully. All users, edits, uploaded CSV files, "
                "and extracted feature state were deleted."
                f" Removed {cleanup_result['deleted_dataset_files']} uploaded dataset file(s)."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
