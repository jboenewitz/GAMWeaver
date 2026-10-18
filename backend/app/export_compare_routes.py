"""Exported model comparison page API."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from .api_common import _read_uploaded_json_artifact, _require_superadmin
from .held_out_overlap import compare_csv_to_artifact_test_sets
from .ml_service import CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN, _json_safe, ml_service
from .models import ModelComparePrepareResponse, ModelComparePreviewRequest, ModelComparePreviewResponse
from .training_overlap import compare_training_to_original_test

router = APIRouter()


@router.post("/api/model-compare/prepare", response_model=ModelComparePrepareResponse)
async def prepare_model_compare(
    http_request: Request,
    left_file: UploadFile = File(...),
    right_file: UploadFile = File(...),
):
    """Validate two model artifacts and return a preview-only comparison payload."""
    try:
        _require_superadmin(http_request)
        left_artifact = await _read_uploaded_json_artifact(left_file)
        right_artifact = await _read_uploaded_json_artifact(right_file)

        left_prepared = ml_service.prepare_model_compare_artifact(
            left_artifact,
            filename=left_file.filename or "left-artifact.json",
        )
        right_prepared = ml_service.prepare_model_compare_artifact(
            right_artifact,
            filename=right_file.filename or "right-artifact.json",
        )

        shared_features = ml_service._validate_model_compare_pair(
            left_prepared,
            right_prepared,
        )
        left_selected_submission_ids = [
            str(submission.get("submission_id"))
            for feature in left_prepared.get("submissions_by_feature") or []
            for submission in feature.get("submissions") or []
            if submission.get("submission_id")
        ]
        right_selected_submission_ids = [
            str(submission.get("submission_id"))
            for feature in right_prepared.get("submissions_by_feature") or []
            for submission in feature.get("submissions") or []
            if submission.get("submission_id")
        ]
        preview = ml_service.build_model_compare_preview(
            left_artifact=left_prepared,
            right_artifact=right_prepared,
            left_selected_submission_ids=left_selected_submission_ids,
            right_selected_submission_ids=right_selected_submission_ids,
            use_confidence=True,
            limit_edited_mean_deviation=False,
            feature_names=shared_features,
        )
        return ModelComparePrepareResponse(
            left_artifact=left_prepared,
            right_artifact=right_prepared,
            shared_features=shared_features,
            preview=ModelComparePreviewResponse(**preview),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model-compare/held-out-overlap")
async def check_model_compare_held_out_overlap(
    http_request: Request,
    original_file: UploadFile = File(...),
    edited_file: UploadFile = File(...),
    csv_file: UploadFile = File(...),
    source_csv_file: Optional[UploadFile] = File(None),
):
    """Report training/test leakage and the separate CSV row comparison."""
    try:
        _require_superadmin(http_request)
        if not (csv_file.filename or "").lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        if source_csv_file and not (source_csv_file.filename or "").lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV source files are supported")
        original_artifact = await _read_uploaded_json_artifact(original_file)
        edited_artifact = await _read_uploaded_json_artifact(edited_file)
        original_prepared = ml_service.prepare_model_compare_artifact(
            original_artifact,
            filename=original_file.filename or "original-artifact.json",
        )
        edited_prepared = ml_service.prepare_model_compare_artifact(
            edited_artifact,
            filename=edited_file.filename or "edited-artifact.json",
        )
        ml_service._validate_model_compare_pair(original_prepared, edited_prepared)
        csv_bytes = await csv_file.read()
        source_bytes = await source_csv_file.read() if source_csv_file else None
        descriptive_overlap = compare_csv_to_artifact_test_sets(
            csv_bytes,
            original_prepared,
            edited_prepared,
            column_aliases=CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN,
        )
        try:
            training_overlap = compare_training_to_original_test(
                original_prepared,
                edited_prepared,
                csv_bytes,
                source_csv_bytes=source_bytes,
                column_aliases=CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN,
            )
        except ValueError as exc:
            training_overlap = {"status": "unavailable", "reason": str(exc)}
        return {**descriptive_overlap, "training_overlap": training_overlap}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/api/model-compare/preview", response_model=ModelComparePreviewResponse)
async def preview_model_compare(
    request: ModelComparePreviewRequest,
    http_request: Request,
):
    """Recompute an isolated comparison preview from prepared artifact payloads."""
    try:
        _require_superadmin(http_request)
        preview = ml_service.build_model_compare_preview(
            left_artifact=_json_safe(request.left_artifact.dict()),
            right_artifact=_json_safe(request.right_artifact.dict()),
            left_selected_submission_ids=list(request.left_selected_submission_ids),
            right_selected_submission_ids=list(request.right_selected_submission_ids),
            use_confidence=bool(request.use_confidence),
            limit_edited_mean_deviation=bool(request.limit_edited_mean_deviation),
            feature_names=list(request.feature_names) if request.feature_names else None,
            overall_mae_feature_names=(
                list(request.overall_mae_feature_names)
                if request.overall_mae_feature_names is not None
                else None
            ),
            mean_deviation_feature_names=(
                list(request.mean_deviation_feature_names)
                if request.mean_deviation_feature_names is not None
                else None
            ),
            mean_deviation_scaled_feature_names=(
                list(request.mean_deviation_scaled_feature_names)
                if request.mean_deviation_scaled_feature_names is not None
                else None
            ),
            manual_feature_shifts=(
                dict(request.manual_feature_shifts)
                if request.manual_feature_shifts is not None
                else None
            ),
            intercept_adjustment=request.intercept_adjustment,
            apply_training_intercept_correction=request.apply_training_intercept_correction,
            feature_metrics_on_original_test_set=request.feature_metrics_on_original_test_set,
            held_out_test_source=(
                str(request.held_out_test_source)
                if request.held_out_test_source is not None
                else "right"
            ),
        )
        return ModelComparePreviewResponse(**preview)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
