"""FastAPI main application for Bike Rental Prediction API."""

from fastapi import FastAPI, HTTPException
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
    UserEditsRequest,
    CombinedEditsResponse,
    ResetDatabaseResponse,
)
from .ml_service import ml_service
from .db_service import db_service

app = FastAPI(
    title="Bike Rental Prediction API",
    description="API for predicting bike rentals using IGANN (Interpretable Generalized Additive Neural Networks)",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def login_or_create_user(request: UserLoginRequest):
    """Login or create a new user."""
    try:
        if not request.name or not request.name.strip():
            raise HTTPException(status_code=400, detail="Name is required")
        
        user = db_service.get_or_create_user(request.name.strip())
        return UserResponse(**user)
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
async def get_combined_predictions_comparison():
    """Get predictions comparison using combined edits from all users."""
    try:
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        # Ensure shape functions are loaded (needed for visualization)
        if not ml_service.original_shape_functions:
            ml_service.get_shape_functions()
        
        # Get combined edits
        combined_edits = db_service.get_combined_edits()
        
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


# ==================== Database Management ====================

@app.post("/api/database/reset", response_model=ResetDatabaseResponse)
async def reset_database():
    """Reset the entire database (users and edits)."""
    try:
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
