"""FastAPI main application for version-scoped dataset prediction workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db_service import db_service
from .ml_service import ml_service
from .models import (
    AdminCreateUserRequest,
    DatasetContextResponse,
    DatasetInspectResponse,
    DatasetSelectionRequest,
    DeleteEditRequest,
    DeleteEditResponse,
    EditedShapeFunctionsRequest,
    InviteCreateResponse,
    ModelMetrics,
    NotificationsResponse,
    PredictionOutput,
    PredictionRequest,
    ResetDatabaseResponse,
    TrainModelRequest,
    TrainModelResponse,
    UserListResponse,
    UserLoginRequest,
    UserPreferencesRequest,
    UserPreferencesResponse,
    UserRegisterRequest,
    UserResponse,
)
from .security import create_admin_token, verify_admin_token


app = FastAPI(
    title="Dataset Prediction API",
    description="API for training persisted IGANN model versions on uploaded datasets.",
    version="2.0.0",
)


def _get_superadmin_payload(request: Request) -> Dict[str, Any]:
    token = request.headers.get("x-superadmin-token", "")
    if not token:
        return {}
    payload = verify_admin_token(token, settings.auth_token_secret)
    if not payload:
        return {}
    if payload.get("sub") != settings.superadmin_username:
        return {}
    return payload


def _require_superadmin(request: Request) -> None:
    if not _get_superadmin_payload(request):
        raise HTTPException(status_code=403, detail="Superadmin access required")


def _require_destructive_access(request: Request) -> None:
    if _get_superadmin_payload(request):
        return
    if not settings.allow_destructive_actions:
        raise HTTPException(status_code=403, detail="Destructive actions are disabled")
    if settings.demo_admin_secret:
        provided = request.headers.get("x-demo-admin-secret", "")
        if provided != settings.demo_admin_secret:
            raise HTTPException(status_code=403, detail="Demo admin secret required")


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_user_or_404(user_id: int) -> Dict[str, Any]:
    user = db_service.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def _get_dataset_or_404(dataset_id: int) -> Dict[str, Any]:
    dataset = db_service.get_dataset_by_id(dataset_id, include_schema=True, include_csv_data=True)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


def _get_model_version_or_404(model_version_id: int, *, include_artifact: bool = False) -> Dict[str, Any]:
    model_version = db_service.get_model_version_by_id(model_version_id, include_artifact=include_artifact)
    if model_version is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return model_version


def _edited_shape_functions_to_dicts(request: EditedShapeFunctionsRequest) -> List[Dict[str, Any]]:
    return [
        {
            "feature_name": sf.feature_name,
            "feature_type": sf.feature_type,
            "edited_points": [
                {
                    "x_value": point.x_value,
                    "y_value": point.y_value,
                    "weight": point.weight,
                    "message": point.message,
                }
                for point in sf.edited_points
            ],
        }
        for sf in request.edited_shape_functions
    ]


def _build_active_context(user_id: int) -> Dict[str, Any]:
    context = db_service.get_active_dataset_context(user_id)
    active_dataset = context["active_dataset"]
    active_model_version = context["active_model_version"]

    data_summary = None
    prediction_fields: List[Dict[str, Any]] = []
    capabilities: Dict[str, Any] = {}

    if active_dataset:
        dataset_record = db_service.get_dataset_by_id(
            active_dataset["id"],
            include_schema=True,
            include_csv_data=True,
        )
        if dataset_record is not None:
            data_summary = ml_service.get_dataset_summary(dataset_record)
            prediction_fields = dataset_record["schema"]["prediction_fields"]
            capabilities = dataset_record["schema"].get("capabilities", {})

    model_status = {
        "dataset_id": active_dataset["id"] if active_dataset else None,
        "model_version_id": active_model_version["id"] if active_model_version else None,
        "dataset_name": active_dataset["display_name"] if active_dataset else None,
        "version_number": active_model_version["version_number"] if active_model_version else None,
        "data_loaded": bool(active_dataset),
        "is_trained": bool(active_model_version),
        "train_size": active_model_version["train_size"] if active_model_version else 0,
        "test_size": active_model_version["test_size"] if active_model_version else 0,
        "features": active_dataset["schema"]["feature_names"] if active_dataset else [],
    }

    return {
        "datasets": context["datasets"],
        "active_dataset": active_dataset,
        "active_model_version": active_model_version,
        "model_status": model_status,
        "data_summary": data_summary,
        "prediction_fields": prediction_fields,
        "capabilities": capabilities,
    }


def _get_saved_display_edits(user_id: int, model_version_id: int) -> List[Dict[str, Any]]:
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    stored = db_service.get_user_edits_as_list(user_id, model_version_id)
    return ml_service.convert_storage_to_display_format(model_version, stored)


def _get_saved_offsets(user_id: int, model_version_id: int) -> Dict[str, Dict[Any, float]]:
    stored = db_service.get_user_edits_as_list(user_id, model_version_id)
    return ml_service.build_offsets_from_storage_edits(stored)


async def _ensure_default_dataset_exists() -> None:
    datasets = db_service.list_datasets()
    if datasets:
        return

    payload = ml_service.build_default_dataset_payload()
    db_service.create_dataset(
        display_name=payload["display_name"],
        original_filename=payload["original_filename"],
        target_column=payload["target_column"],
        schema_json=payload["schema"],
        csv_data=payload["csv_data"],
        uploaded_by_user_id=None,
    )


async def _migrate_legacy_edits() -> None:
    edit_count, notification_count = db_service.get_unscoped_edit_counts()
    if edit_count == 0 and notification_count == 0:
        return

    datasets = db_service.list_datasets()
    if not datasets:
        return

    default_dataset = _get_dataset_or_404(datasets[0]["id"])
    model_version = db_service.get_latest_model_version_for_dataset(default_dataset["id"])
    if model_version is None:
        trained = ml_service.train_model(default_dataset)
        model_version = db_service.create_model_version(
            dataset_id=default_dataset["id"],
            training_params=trained["training_params"],
            train_size=trained["train_size"],
            test_size=trained["test_size"],
            schema_snapshot=trained["schema_snapshot"],
            metrics_json=trained["metrics"],
            artifact_blob=trained["artifact_blob"],
            created_by_user_id=None,
        )
        ml_service.prime_artifact_cache(model_version["id"], trained["artifact"])

    db_service.assign_legacy_model_version(model_version["id"])


@app.on_event("startup")
async def initialize_app_state():
    db_service.ensure_superadmin()
    await _ensure_default_dataset_exists()
    await _migrate_legacy_edits()


@app.get("/")
async def root():
    return {
        "message": "Dataset Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "dataset_count": len(db_service.list_datasets()),
    }


@app.get("/health")
async def health_check_root():
    return {"status": "healthy"}


# ==================== Dataset Context Endpoints ====================

@app.get("/api/datasets")
async def list_datasets():
    return {"datasets": db_service.list_datasets()}


@app.get("/api/users/{user_id}/context", response_model=DatasetContextResponse)
async def get_active_context(user_id: int):
    _get_user_or_404(user_id)
    return _build_active_context(user_id)


@app.post("/api/users/{user_id}/active-dataset", response_model=DatasetContextResponse)
async def set_active_dataset(user_id: int, request: DatasetSelectionRequest):
    _get_user_or_404(user_id)
    dataset = db_service.get_dataset_by_id(request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db_service.set_active_dataset(user_id, request.dataset_id)
    return _build_active_context(user_id)


@app.post("/api/admin/datasets/inspect", response_model=DatasetInspectResponse)
async def inspect_dataset_upload(
    http_request: Request,
    file: UploadFile = File(...),
):
    _require_superadmin(http_request)
    csv_bytes = await file.read()
    try:
        return ml_service.inspect_dataset_upload(csv_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/admin/datasets")
async def upload_dataset(
    http_request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    display_name: str = Form(""),
    uploaded_by_user_id: Optional[int] = Form(default=None),
):
    _require_superadmin(http_request)
    csv_bytes = await file.read()

    try:
        schema = ml_service.build_dataset_schema(csv_bytes, target_column.strip())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if uploaded_by_user_id is not None:
        _get_user_or_404(uploaded_by_user_id)

    cleaned_display_name = display_name.strip() or (file.filename or "Uploaded Dataset")
    dataset = db_service.create_dataset(
        display_name=cleaned_display_name,
        original_filename=file.filename or cleaned_display_name,
        target_column=target_column.strip(),
        schema_json=schema,
        csv_data=csv_bytes,
        uploaded_by_user_id=uploaded_by_user_id,
    )

    if uploaded_by_user_id is not None:
        db_service.set_active_dataset(uploaded_by_user_id, dataset["id"])

    return {"dataset": dataset}


# ==================== Model Version Endpoints ====================

@app.post("/api/datasets/{dataset_id}/train", response_model=TrainModelResponse)
async def train_dataset(dataset_id: int, request: TrainModelRequest):
    _get_user_or_404(request.user_id)
    dataset = _get_dataset_or_404(dataset_id)

    try:
        trained = ml_service.train_model(
            dataset,
            n_estimators=request.n_estimators,
            test_size=request.test_size,
            random_state=request.random_state,
        )
        model_version = db_service.create_model_version(
            dataset_id=dataset_id,
            training_params=trained["training_params"],
            train_size=trained["train_size"],
            test_size=trained["test_size"],
            schema_snapshot=trained["schema_snapshot"],
            metrics_json=trained["metrics"],
            artifact_blob=trained["artifact_blob"],
            created_by_user_id=request.user_id,
        )
        ml_service.prime_artifact_cache(model_version["id"], trained["artifact"])
        db_service.set_active_model_version(request.user_id, dataset_id, model_version["id"])

        return TrainModelResponse(
            success=True,
            message="Model trained successfully",
            metrics=ModelMetrics(**trained["metrics"]),
            model_version_id=model_version["id"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-versions/{model_version_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_version_id: int):
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        metrics = ml_service.evaluate_model(model_version)
        return ModelMetrics(**metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-versions/{model_version_id}/shape-functions")
async def get_shape_functions(model_version_id: int):
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        shape_functions = ml_service.get_shape_functions(model_version)
        return {"shape_functions": shape_functions}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-versions/{model_version_id}/predictions-vs-actual")
async def get_predictions_vs_actual(model_version_id: int):
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        return ml_service.get_predictions_vs_actual(model_version)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-versions/{model_version_id}/predictions-comparison")
async def get_predictions_comparison(model_version_id: int, user_id: int = Query(...)):
    _get_user_or_404(user_id)
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        offsets = _get_saved_offsets(user_id, model_version_id)
        return ml_service.get_predictions_comparison(model_version, offsets)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/model-versions/{model_version_id}/preview-comparison")
async def preview_predictions_comparison(
    model_version_id: int,
    request: EditedShapeFunctionsRequest,
    user_id: int = Query(...),
):
    _get_user_or_404(user_id)
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)

    try:
        saved_display_edits = _get_saved_display_edits(user_id, model_version_id)
        overlay_edits = _edited_shape_functions_to_dicts(request)
        merged_display_edits = ml_service.merge_display_edits(saved_display_edits, overlay_edits)
        merged_storage_edits = ml_service.convert_edits_for_storage(model_version, merged_display_edits)
        offsets = ml_service.build_offsets_from_storage_edits(merged_storage_edits)
        return ml_service.get_predictions_comparison(model_version, offsets)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/model-versions/{model_version_id}/predict", response_model=PredictionOutput)
async def predict(model_version_id: int, request: PredictionRequest):
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        prediction = ml_service.predict(model_version, request.input_features)
        return PredictionOutput(
            predicted_value=max(0.0, prediction),
            input_features=request.input_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/datasets/{dataset_id}/hourly-pattern")
async def get_hourly_pattern(dataset_id: int):
    dataset = _get_dataset_or_404(dataset_id)
    try:
        return ml_service.get_hourly_pattern(dataset)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ==================== Version-Scoped User Edit Endpoints ====================

@app.get("/api/users/{user_id}/model-versions/{model_version_id}/edits")
async def get_user_edits(user_id: int, model_version_id: int):
    _get_user_or_404(user_id)
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        storage_edits = db_service.get_user_edits_as_list(user_id, model_version_id)
        display_edits = ml_service.convert_storage_to_display_format(model_version, storage_edits)
        return {"user_id": user_id, "model_version_id": model_version_id, "edits": display_edits}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/users/{user_id}/model-versions/{model_version_id}/edits")
async def save_user_edits(user_id: int, model_version_id: int, request: EditedShapeFunctionsRequest):
    _get_user_or_404(user_id)
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        edited_shape_functions = _edited_shape_functions_to_dicts(request)
        storage_format = ml_service.convert_edits_for_storage(model_version, edited_shape_functions)
        db_service.save_user_edits(user_id, model_version_id, storage_format)
        return {"success": True, "message": "Edits saved successfully"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/users/{user_id}/model-versions/{model_version_id}/edits")
async def clear_user_edits(user_id: int, model_version_id: int):
    _get_user_or_404(user_id)
    _get_model_version_or_404(model_version_id)
    try:
        db_service.clear_user_edits(user_id, model_version_id)
        return {"success": True, "message": "User edits cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/users/{user_id}/model-versions/{model_version_id}/edits/{feature_name}")
async def clear_user_feature_edits(user_id: int, model_version_id: int, feature_name: str):
    _get_user_or_404(user_id)
    _get_model_version_or_404(model_version_id)
    try:
        db_service.clear_user_feature_edits(user_id, model_version_id, feature_name)
        return {"success": True, "message": f"Edits for {feature_name} cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ==================== Version-Scoped Combined Results Endpoints ====================

@app.get("/api/model-versions/{model_version_id}/combined/edits")
async def get_combined_edits(model_version_id: int):
    _get_model_version_or_404(model_version_id)
    try:
        return db_service.get_combined_edits_detailed(model_version_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-versions/{model_version_id}/combined/edit-logs")
async def get_edit_logs(model_version_id: int):
    _get_model_version_or_404(model_version_id)
    try:
        return db_service.get_edit_logs(model_version_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-versions/{model_version_id}/combined/users")
async def get_users_with_edits(model_version_id: int):
    _get_model_version_or_404(model_version_id)
    try:
        users = db_service.get_users_with_edits(model_version_id)
        return {"users": users}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-versions/{model_version_id}/combined/predictions-comparison")
async def get_combined_predictions_comparison(model_version_id: int, weighted: bool = True):
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        combined_offsets = db_service.get_combined_edits(model_version_id, weighted=weighted)
        comparison = ml_service.get_predictions_comparison(model_version, combined_offsets)
        comparison["total_users_with_edits"] = db_service.get_combined_edits_detailed(model_version_id)[
            "total_users_with_edits"
        ]
        comparison["combined_shape_functions"] = db_service.get_combined_edits_detailed(model_version_id)[
            "shape_functions"
        ]
        comparison["original_shape_functions"] = ml_service.get_shape_functions(model_version)
        comparison["combined_shape_functions_display"] = ml_service.apply_offsets_to_shape_functions(
            model_version,
            combined_offsets,
        )
        return comparison
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-versions/{model_version_id}/combined/per-user-shape-functions")
async def get_per_user_shape_functions(model_version_id: int, weighted: bool = True):
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        users_with_edits = db_service.get_users_with_edits(model_version_id)
        result = []
        for user in users_with_edits:
            user_edits_raw = db_service.get_user_edits_raw(user["id"], model_version_id)
            offsets: Dict[str, Dict[Any, float]] = {}
            for feature_name, points in user_edits_raw.items():
                offsets[feature_name] = {}
                for x_value, point in points.items():
                    raw_offset = point["offset"]
                    weight_value = point["weight"]
                    offsets[feature_name][x_value] = raw_offset * weight_value if weighted else raw_offset

            result.append(
                {
                    "user_id": user["id"],
                    "user_name": user["name"],
                    "shape_functions": ml_service.apply_offsets_to_shape_functions(model_version, offsets),
                }
            )
        return {"users": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/model-versions/{model_version_id}/combined/predict", response_model=PredictionOutput)
async def combined_predict(model_version_id: int, request: PredictionRequest):
    model_version = _get_model_version_or_404(model_version_id, include_artifact=True)
    try:
        combined_offsets = db_service.get_combined_edits(model_version_id)
        prediction = ml_service.predict(model_version, request.input_features, combined_offsets)
        return PredictionOutput(
            predicted_value=max(0.0, prediction),
            input_features=request.input_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ==================== User/Auth Endpoints ====================

@app.post("/api/users/login", response_model=UserResponse)
@app.post("/api/auth/login", response_model=UserResponse)
async def login_user(request: UserLoginRequest):
    try:
        username = request.username.strip()
        if not username or not request.password:
            raise HTTPException(status_code=400, detail="Username and password are required")

        if username == settings.superadmin_username:
            if not settings.superadmin_password:
                raise HTTPException(status_code=500, detail="Superadmin password not configured")
            if request.password != settings.superadmin_password:
                raise HTTPException(status_code=401, detail="Invalid credentials")

            user = db_service.ensure_superadmin()
            if user is None:
                raise HTTPException(status_code=500, detail="Superadmin not initialized")

            token = create_admin_token(
                settings.superadmin_username,
                settings.auth_token_secret,
                settings.admin_token_ttl_hours * 3600,
            )
            return UserResponse(**user, access_token=token)

        user = db_service.verify_user_credentials(username, request.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return UserResponse(**user)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/auth/register", response_model=UserResponse)
async def register_user(request: UserRegisterRequest):
    try:
        username = request.username.strip()
        if not username or not request.password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        if not request.invite_token or not request.invite_token.strip():
            raise HTTPException(status_code=400, detail="Invite token is required")
        if db_service.get_user_by_name(username):
            raise HTTPException(status_code=409, detail="User already exists")
        if not db_service.consume_invite_token(request.invite_token.strip()):
            raise HTTPException(status_code=403, detail="Invalid or expired invite token")

        user = db_service.create_user_with_password(username, request.password)
        return UserResponse(**user, is_new=True)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/users", response_model=UserListResponse)
async def get_all_users():
    try:
        users = db_service.get_all_users()
        return UserListResponse(users=users)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/admin/users", response_model=UserListResponse)
async def get_all_users_admin(request: Request):
    try:
        _require_superadmin(request)
        users = db_service.get_all_users()
        return UserListResponse(users=users)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/admin/users", response_model=UserResponse)
async def create_user_admin(request_body: AdminCreateUserRequest, request: Request):
    try:
        _require_superadmin(request)
        username = request_body.username.strip()
        if not username or not request_body.password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        user = db_service.create_user_with_password(username, request_body.password)
        return UserResponse(**user)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/admin/invites", response_model=InviteCreateResponse)
async def create_invite(request: Request):
    try:
        _require_superadmin(request)
        admin_user = db_service.get_user_by_name(settings.superadmin_username)
        admin_id = admin_user["id"] if admin_user else None
        invite = db_service.create_invite_token(admin_id)
        return InviteCreateResponse(**invite)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ==================== Edit Deletion and Notifications ====================

@app.post("/api/edits/delete", response_model=DeleteEditResponse)
async def delete_edit(request: DeleteEditRequest, http_request: Request):
    try:
        _require_destructive_access(http_request)
        if not request.reason or not request.reason.strip():
            raise HTTPException(status_code=400, detail="Reason is required")
        success = db_service.delete_edit(
            edit_id=request.edit_id,
            deleted_by_user_id=request.deleted_by_user_id,
            reason=request.reason.strip(),
        )
        if not success:
            raise HTTPException(status_code=404, detail="Edit not found")
        return DeleteEditResponse(success=True, message="Edit deleted successfully")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/users/{user_id}/notifications", response_model=NotificationsResponse)
async def get_user_notifications(user_id: int, model_version_id: Optional[int] = Query(default=None)):
    _get_user_or_404(user_id)
    try:
        notifications = db_service.get_unseen_notifications(user_id, model_version_id=model_version_id)
        return NotificationsResponse(notifications=notifications)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/users/{user_id}/notifications/mark-seen")
async def mark_notifications_seen(user_id: int, model_version_id: Optional[int] = Query(default=None)):
    _get_user_or_404(user_id)
    try:
        db_service.mark_notifications_seen(user_id, model_version_id=model_version_id)
        return {"success": True, "message": "Notifications marked as seen"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ==================== User Preferences ====================

@app.get("/api/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(user_id: int):
    _get_user_or_404(user_id)
    try:
        prefs = db_service.get_user_preferences(user_id)
        return UserPreferencesResponse(preferences=prefs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(user_id: int, request: UserPreferencesRequest):
    _get_user_or_404(user_id)
    try:
        updated = db_service.update_user_preferences(user_id, request.preferences)
        return UserPreferencesResponse(preferences=updated)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ==================== Database Management ====================

@app.post("/api/database/reset", response_model=ResetDatabaseResponse)
async def reset_database(http_request: Request):
    try:
        _require_destructive_access(http_request)
        db_service.reset_all_data()
        db_service.ensure_superadmin()
        await _ensure_default_dataset_exists()
        return ResetDatabaseResponse(
            success=True,
            message="Database reset successfully. All users and edits have been deleted.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
