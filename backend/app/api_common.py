"""Authentication and request helpers shared by the API routers."""

import json
from typing import Any, Dict

from fastapi import HTTPException, Request, UploadFile

from .config import settings
from .security import verify_admin_token


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


def _normalize_required_profession(raw_profession: str) -> str:
    profession = raw_profession.strip()
    if not profession:
        raise HTTPException(status_code=400, detail="Profession is required")
    return profession


def _extract_prediction_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept both direct feature dictionaries and {"features": {...}} payloads.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid prediction payload")
    if "features" in payload:
        features = payload.get("features")
        if not isinstance(features, dict):
            raise HTTPException(status_code=400, detail="'features' must be an object")
        return features
    return payload


async def _read_uploaded_json_artifact(file: UploadFile) -> Dict[str, Any]:
    file_name = (file.filename or "").strip()
    if file_name and not file_name.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON model artifacts are supported")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse JSON artifact: {exc}",
        ) from exc


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
