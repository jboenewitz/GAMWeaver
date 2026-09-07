"""Helpers for representing an untouched shape function as an explicit no-op."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4


NOOP_EDIT_MESSAGE = "Let the chart be"
NOOP_EDIT_WEIGHT = 0.5


def encode_numeric_storage_key(value: Any) -> str:
    """Match the stable numeric-key format used by :mod:`ml_service`."""
    return f"x:{float(value):.12g}"


def build_imputed_noop_submission(
    shape_function: Dict[str, Any],
    *,
    submission_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete zero-offset submission for one base shape function."""
    feature_name = str(shape_function["feature_name"])
    feature_type = str(shape_function["feature_type"])
    if feature_type not in {"numeric", "categorical"}:
        raise ValueError(f"Unsupported feature type for no-op '{feature_name}'")

    points = []
    for x_value in shape_function.get("x_values") or []:
        points.append(
            {
                "x_value": (
                    str(x_value)
                    if feature_type == "categorical"
                    else encode_numeric_storage_key(x_value)
                ),
                "y_value": 0.0,
                "weight": NOOP_EDIT_WEIGHT,
                "message": NOOP_EDIT_MESSAGE,
            }
        )

    return {
        "feature_name": feature_name,
        "feature_type": feature_type,
        "submission_id": submission_id or uuid4().hex,
        "message": NOOP_EDIT_MESSAGE,
        "sureness": 5,
        "is_imputed_noop": True,
        "edited_points": points,
    }


def complete_shape_function_edits_export(
    payload: Dict[str, Any],
    shape_functions: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return an edits payload with one submission per participant and feature.

    A participant is a declared user with at least one non-no-op submission.
    Existing entries are preserved; only absent user/feature pairs receive an
    imputed zero-offset submission.  This makes the operation idempotent.
    """
    completed = deepcopy(payload or {})
    if not completed.get("included"):
        return completed

    shape_by_name = {
        str(shape["feature_name"]): shape
        for shape in shape_functions
        if isinstance(shape, dict) and shape.get("feature_name")
    }
    if not shape_by_name:
        return completed

    user_groups = [
        group for group in completed.get("edits", []) if isinstance(group, dict)
    ]
    group_by_user = {
        str(group.get("user_name", "")).strip(): group
        for group in user_groups
        if str(group.get("user_name", "")).strip()
    }
    participating_users = {
        user_name
        for user_name, group in group_by_user.items()
        if any(
            isinstance(shape, dict) and not shape.get("is_imputed_noop", False)
            for shape in group.get("shape_functions", [])
        )
    }

    for user_name in participating_users:
        group = group_by_user[user_name]
        existing_features = {
            str(shape.get("feature_name", "")).strip()
            for shape in group.get("shape_functions", [])
            if isinstance(shape, dict)
        }
        for feature_name, base_shape in shape_by_name.items():
            if feature_name not in existing_features:
                group.setdefault("shape_functions", []).append(
                    build_imputed_noop_submission(base_shape)
                )

    completed["edits"] = [
        group
        for group in user_groups
        if str(group.get("user_name", "")).strip() in participating_users
    ]
    completed["users"] = [
        user
        for user in completed.get("users", [])
        if isinstance(user, dict)
        and str(user.get("name", "")).strip() in participating_users
    ]
    completed["user_count"] = len(participating_users)
    completed["submission_count"] = sum(
        len(group.get("shape_functions", []))
        for group in completed["edits"]
    )
    return completed
