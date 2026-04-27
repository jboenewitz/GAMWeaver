"""FastAPI main application for Bike Rental Prediction API."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from .models import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    ModelMetrics,
    ShapeFunctionData,
    DataSummary,
    TrainModelRequest,
    TrainModelResponse,
    EditedShapeFunctionsRequest,
    PredictionComparisonResponse,
    ComparisonMetrics,
    UserLoginRequest,
    UserResponse,
    UserListResponse,
    UserRegisterRequest,
    AdminCreateUserRequest,
    InviteCreateResponse,
    UserEditsRequest,
    CombinedEditsResponse,
    ResetDatabaseResponse,
    DeleteEditRequest,
    DeleteEditResponse,
    NotificationsResponse,
    UserPreferencesRequest,
    UserPreferencesResponse,
)
from .ml_service import ml_service
from .db_service import db_service
from .config import settings
from .security import create_admin_token, verify_admin_token

app = FastAPI(
    title="Bike Rental Prediction API",
    description="API for predicting bike rentals using IGANN (Interpretable Generalized Additive Neural Networks)",
    version="1.0.0",
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
    # TODO: Replace demo admin secret with real authentication/authorization.
    if _get_superadmin_payload(request):
        return
    if not settings.allow_destructive_actions:
        raise HTTPException(status_code=403, detail="Destructive actions are disabled")
    if settings.demo_admin_secret:
        provided = request.headers.get("x-demo-admin-secret", "")
        if provided != settings.demo_admin_secret:
            raise HTTPException(status_code=403, detail="Demo admin secret required")


# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def ensure_superadmin_user():
    db_service.ensure_superadmin()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bike Rental Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "load_data": "/api/data/load",
            "train_model": "/api/model/train",
            "predict": "/api/predict",
            "shape_functions": "/api/model/shape-functions",
            "metrics": "/api/model/metrics",
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_trained": ml_service.is_trained}


@app.get("/health")
async def health_check_root():
    """Health check endpoint for platform probes."""
    return {"status": "healthy"}


@app.post("/api/data/load")
async def load_data():
    """Load and prepare the dataset."""
    try:
        result = ml_service.load_data()
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/summary")
async def get_data_summary():
    """Get summary statistics of the dataset."""
    try:
        summary = ml_service.get_data_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/distributions")
async def get_feature_distributions():
    """Get feature distributions for visualization."""
    try:
        distributions = ml_service.get_feature_distributions()
        return distributions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/hourly-pattern")
async def get_hourly_pattern():
    """Get hourly bike rental pattern."""
    try:
        pattern = ml_service.get_hourly_pattern()
        return pattern
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/train", response_model=TrainModelResponse)
async def train_model(request: TrainModelRequest = None):
    """Train the IGANN model."""
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """Get model performance metrics."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        metrics = ml_service.evaluate_model()
        return ModelMetrics(**metrics)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/shape-functions")
async def get_shape_functions():
    """Get shape function data for all features."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        shape_functions = ml_service.get_shape_functions()
        return {"shape_functions": shape_functions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/predictions-vs-actual")
async def get_predictions_vs_actual():
    """Get predictions vs actual values for visualization."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        data = ml_service.get_predictions_vs_actual()
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a single bike rental prediction."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet. Call /api/model/train first.")
        
        features = {
            "temperature": input_data.temperature,
            "humidity": input_data.humidity,
            "windspeed": input_data.windspeed,
            "time_of_day": input_data.time_of_day,
            "type_of_day": input_data.type_of_day,
            "weathersituation": input_data.weathersituation,
        }
        
        prediction = ml_service.predict(features)
        
        return PredictionOutput(
            predicted_count=max(0, prediction),  # Ensure non-negative
            input_features=features
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch", response_model=BatchPredictionOutput)
async def batch_predict(input_data: BatchPredictionInput):
    """Make batch predictions."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        features_list = [
            {
                "temperature": p.temperature,
                "humidity": p.humidity,
                "windspeed": p.windspeed,
                "time_of_day": p.time_of_day,
                "type_of_day": p.type_of_day,
                "weathersituation": p.weathersituation,
            }
            for p in input_data.predictions
        ]
        
        predictions = ml_service.batch_predict(features_list)
        predictions = [max(0, p) for p in predictions]  # Ensure non-negative
        
        return BatchPredictionOutput(predictions=predictions)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/status")
async def get_model_status():
    """Get the current status of the model."""
    return {
        "is_trained": ml_service.is_trained,
        "data_loaded": ml_service.X_train is not None,
        "features": ml_service.feature_names if ml_service.feature_names else [],
        "train_size": len(ml_service.X_train) if ml_service.X_train is not None else 0,
        "test_size": len(ml_service.X_test) if ml_service.X_test is not None else 0,
    }


@app.post("/api/model/update-shape-functions")
async def update_shape_functions(request: EditedShapeFunctionsRequest):
    """Update shape functions with user edits."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        # Convert to dict format
        edited_sfs = [
            {
                "feature_name": sf.feature_name,
                "feature_type": sf.feature_type,
                "edited_points": [
                    {"x_value": p.x_value, "y_value": p.y_value}
                    for p in sf.edited_points
                ]
            }
            for sf in request.edited_shape_functions
        ]
        
        ml_service.update_shape_functions(edited_sfs)
        return {"success": True, "message": "Shape functions updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/predictions-comparison")
async def get_predictions_comparison():
    """Get comparison between original and interactive predictions."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        comparison = ml_service.get_predictions_comparison()
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/reset-shape-functions")
async def reset_shape_functions():
    """Reset shape function edits to original values."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        ml_service.shape_function_offsets = {}
        return {"success": True, "message": "Shape functions reset to original"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== User Endpoints ====================

@app.post("/api/users/login", response_model=UserResponse)
@app.post("/api/auth/login", response_model=UserResponse)
async def login_user(request: UserLoginRequest):
    """Login a user with username/password."""
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/register", response_model=UserResponse)
async def register_user(request: UserRegisterRequest):
    """Register a new user via invite token."""
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
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users", response_model=UserListResponse)
async def get_all_users():
    """Get all users."""
    try:
        users = db_service.get_all_users()
        return UserListResponse(users=users)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/users", response_model=UserListResponse)
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


@app.post("/api/admin/users", response_model=UserResponse)
async def create_user_admin(request_body: AdminCreateUserRequest, request: Request):
    """Create a user (superadmin only)."""
    try:
        _require_superadmin(request)
        username = request_body.username.strip()
        if not username or not request_body.password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        user = db_service.create_user_with_password(username, request_body.password)
        return UserResponse(**user)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/invites", response_model=InviteCreateResponse)
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


@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Get a specific user by ID."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/edits")
async def get_user_edits(user_id: int):
    """Get all edits for a specific user."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        edits = db_service.get_user_edits_as_list(user_id)
        return {"user_id": user_id, "edits": edits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users/{user_id}/edits")
async def save_user_edits(user_id: int, request: EditedShapeFunctionsRequest):
    """Save shape function edits for a specific user."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        # Convert to dict format (including weight and message)
        edited_sfs = [
            {
                "feature_name": sf.feature_name,
                "feature_type": sf.feature_type,
                "edited_points": [
                    {"x_value": p.x_value, "y_value": p.y_value, "weight": p.weight, "message": p.message}
                    for p in sf.edited_points
                ]
            }
            for sf in request.edited_shape_functions
        ]
        
        # Update the ML service offsets for immediate feedback
        ml_service.update_shape_functions(edited_sfs)
        
        # Convert to storage format (with indices and offsets) before saving to DB
        storage_format = ml_service.convert_edits_for_storage(edited_sfs)
        db_service.save_user_edits(user_id, storage_format)
        
        return {"success": True, "message": "Edits saved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/users/{user_id}/edits")
async def clear_user_edits(user_id: int):
    """Clear all edits for a specific user."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        db_service.clear_user_edits(user_id)
        ml_service.shape_function_offsets = {}
        
        return {"success": True, "message": "User edits cleared"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/users/{user_id}/edits/{feature_name}")
async def clear_user_feature_edits(user_id: int, feature_name: str):
    """Clear edits for a specific user and feature."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        db_service.clear_user_feature_edits(user_id, feature_name)
        
        # Clear the specific feature from offsets if it exists
        if feature_name in ml_service.shape_function_offsets:
            del ml_service.shape_function_offsets[feature_name]
        
        return {"success": True, "message": f"Edits for {feature_name} cleared"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users/{user_id}/load-edits")
async def load_user_edits_to_model(user_id: int):
    """Load a user's saved edits into the ML service for visualization."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        # Get stored edits (with indices and offsets)
        storage_edits = db_service.get_user_edits_as_list(user_id)
        
        if storage_edits:
            # Apply to ML service for predictions
            ml_service.shape_function_offsets = {}
            for sf in storage_edits:
                feature_name = sf["feature_name"]
                feature_type = sf["feature_type"]
                ml_service.shape_function_offsets[feature_name] = {}
                for point in sf["edited_points"]:
                    x_val = point["x_value"]
                    offset = point["y_value"]
                    if feature_type == "categorical":
                        ml_service.shape_function_offsets[feature_name][str(x_val)] = offset
                    else:
                        try:
                            ml_service.shape_function_offsets[feature_name][int(x_val)] = offset
                        except ValueError:
                            pass
            
            # Convert to display format for frontend
            display_edits = ml_service.convert_storage_to_display_format(storage_edits)
        else:
            ml_service.shape_function_offsets = {}
            display_edits = []
        
        return {"success": True, "message": "User edits loaded", "edits": display_edits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Combined Results Endpoints ====================

@app.get("/api/combined/edits")
async def get_combined_edits():
    """Get combined edits from all users."""
    try:
        combined = db_service.get_combined_edits_detailed()
        return combined
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/combined/edit-logs")
async def get_edit_logs():
    """Get detailed edit logs for all users, grouped by feature."""
    try:
        logs = db_service.get_edit_logs()
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/combined/users")
async def get_users_with_edits():
    """Get all users who have made edits."""
    try:
        users = db_service.get_users_with_edits()
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/combined/predictions-comparison")
async def get_combined_predictions_comparison(weighted: bool = True):
    """Get predictions comparison using combined edits from all users.
    weighted=true (default): confidence-weighted average of user offsets.
    weighted=false: simple unweighted mean of user offsets.
    """
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")

        # Ensure shape functions are loaded (needed for visualization)
        if not ml_service.original_shape_functions:
            ml_service.get_shape_functions()

        # Get combined edits (weighted or unweighted)
        combined_edits = db_service.get_combined_edits(weighted=weighted)
        
        # Temporarily apply combined edits
        original_offsets = ml_service.shape_function_offsets.copy()
        ml_service.shape_function_offsets = combined_edits
        
        # Get comparison
        comparison = ml_service.get_predictions_comparison()
        
        # Restore original offsets
        ml_service.shape_function_offsets = original_offsets
        
        # Add combined edit info
        combined_details = db_service.get_combined_edits_detailed()
        comparison["total_users_with_edits"] = combined_details["total_users_with_edits"]
        comparison["combined_shape_functions"] = combined_details["shape_functions"]
        
        # Add original shape functions for visualization
        comparison["original_shape_functions"] = list(ml_service.original_shape_functions.values())
        
        # Add combined (modified) shape functions for visualization
        combined_shape_functions_display = []
        for sf in ml_service.original_shape_functions.values():
            feature_name = sf["feature_name"]
            x_values = sf["x_values"]
            original_y = sf["y_values"]
            feature_type = sf["feature_type"]
            
            # Apply combined offsets
            modified_y = []
            for i, (x, y) in enumerate(zip(x_values, original_y)):
                offset = 0.0
                if feature_name in combined_edits:
                    if feature_type == "categorical":
                        offset = combined_edits[feature_name].get(str(x), 0.0)
                    else:
                        offset = combined_edits[feature_name].get(i, 0.0)
                modified_y.append(y + offset)
            
            combined_shape_functions_display.append({
                "feature_name": feature_name,
                "feature_type": feature_type,
                "x_values": x_values,
                "y_values": modified_y,
                "original_y_values": original_y
            })
        
        comparison["combined_shape_functions_display"] = combined_shape_functions_display
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/combined/per-user-shape-functions")
async def get_per_user_shape_functions(weighted: bool = True):
    """Get shape functions for each user who has made edits, applied individually.

    When weighted=True each user's offset is scaled by their confidence weight before
    being added to the original, mirroring how the combined weighted average is built.
    When weighted=False the raw offsets are applied (unweighted).
    """
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")

        if not ml_service.original_shape_functions:
            ml_service.get_shape_functions()

        users_with_edits = db_service.get_users_with_edits()

        result = []
        for user in users_with_edits:
            user_id = user["id"]
            user_name = user["name"]
            # {feature_name: {x_value_str: {offset, weight}}}
            user_edits_raw = db_service.get_user_edits_raw(user_id)

            user_shape_functions = []
            for sf in ml_service.original_shape_functions.values():
                feature_name = sf["feature_name"]
                x_values = sf["x_values"]
                original_y = sf["y_values"]
                feature_type = sf["feature_type"]

                modified_y = []
                for i, (x, y) in enumerate(zip(x_values, original_y)):
                    offset = 0.0
                    if feature_name in user_edits_raw:
                        feat_edits = user_edits_raw[feature_name]
                        if feature_type == "categorical":
                            point = feat_edits.get(str(x))
                        else:
                            # For numeric features, x_value in DB is the index stored as string
                            point = feat_edits.get(str(i))
                        if point:
                            raw_offset = point["offset"]
                            weight_val = point["weight"]
                            offset = raw_offset * weight_val if weighted else raw_offset
                    modified_y.append(float(y) + offset)

                user_shape_functions.append({
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "x_values": x_values,
                    "y_values": modified_y,
                })

            result.append({
                "user_id": user_id,
                "user_name": user_name,
                "shape_functions": user_shape_functions,
            })

        return {"users": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Edit Deletion Endpoints ====================

@app.post("/api/edits/delete", response_model=DeleteEditResponse)
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


@app.get("/api/users/{user_id}/notifications")
async def get_user_notifications(user_id: int):
    """Get unseen deletion notifications for a user."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        notifications = db_service.get_unseen_notifications(user_id)
        return {"notifications": notifications}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users/{user_id}/notifications/mark-seen")
async def mark_notifications_seen(user_id: int):
    """Mark all notifications for a user as seen."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        db_service.mark_notifications_seen(user_id)
        return {"success": True, "message": "Notifications marked as seen"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(user_id: int):
    """Get stored UI preferences for a user."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        prefs = db_service.get_user_preferences(user_id)
        return UserPreferencesResponse(preferences=prefs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(user_id: int, request: UserPreferencesRequest):
    """Update (merge) UI preferences for a user."""
    try:
        user = db_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        updated = db_service.update_user_preferences(user_id, request.preferences)
        return UserPreferencesResponse(preferences=updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Combined Predict Endpoint ====================

@app.post("/api/combined/predict", response_model=PredictionOutput)
async def combined_predict(input_data: PredictionInput):
    """Make a prediction using the combined shape function edits from all users."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet.")
        
        # Ensure shape functions are loaded
        if not ml_service.original_shape_functions:
            ml_service.get_shape_functions()
        
        # Get combined edits from all users
        combined_offsets = db_service.get_combined_edits()
        
        features = {
            "temperature": input_data.temperature,
            "humidity": input_data.humidity,
            "windspeed": input_data.windspeed,
            "time_of_day": input_data.time_of_day,
            "type_of_day": input_data.type_of_day,
            "weathersituation": input_data.weathersituation,
        }
        
        prediction = ml_service.predict_with_offsets(features, combined_offsets)
        
        return PredictionOutput(
            predicted_count=max(0, prediction),
            input_features=features
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Database Management ====================

@app.post("/api/database/reset", response_model=ResetDatabaseResponse)
async def reset_database(http_request: Request):
    """Reset the entire database (users and edits)."""
    try:
        _require_destructive_access(http_request)
        db_service.reset_all_data()
        ml_service.shape_function_offsets = {}
        return ResetDatabaseResponse(
            success=True,
            message="Database reset successfully. All users and edits have been deleted."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
