"""Database service layer for user and edit management."""

from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import secrets

from .database import (
    SessionLocal,
    User,
    ShapeFunctionEdit,
    DeletedEditNotification,
    InviteToken,
    reset_database as db_reset,
)
from .config import settings
from .security import hash_password, verify_password
from .ml_service import ml_service
from uuid import uuid4


def _decode_numeric_x_for_display(raw_value: Any) -> Any:
    """Best-effort numeric x-value decoding for API display endpoints."""
    if raw_value is None:
        return raw_value
    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    raw_str = str(raw_value).strip()
    if raw_str.startswith("x:"):
        try:
            return float(raw_str[2:])
        except ValueError:
            return raw_value

    try:
        return float(raw_str)
    except ValueError:
        return raw_value


def _format_numeric_display(value: Any) -> str:
    """Format a numeric x-value for compact human-readable summaries."""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _build_x_summary_from_rows(rows: List[ShapeFunctionEdit]) -> str:
    """Build a concise summary of x-values for a submission batch."""
    if not rows:
        return ""

    if rows[0].feature_type == "categorical":
        labels = [str(_decode_numeric_x_for_display(row.x_value)) for row in rows]
        unique_labels = list(dict.fromkeys(labels))
        if len(unique_labels) <= 3:
            return ", ".join(unique_labels)
        preview = ", ".join(unique_labels[:3])
        return f"{preview}, +{len(unique_labels) - 3} more"

    numeric_values = [
        _decode_numeric_x_for_display(row.x_value)
        for row in rows
    ]
    numeric_values = [
        float(value)
        for value in numeric_values
        if isinstance(value, (int, float))
    ]
    if not numeric_values:
        return str(rows[0].x_value)

    numeric_values.sort()
    if len(numeric_values) == 1:
        return _format_numeric_display(numeric_values[0])
    if len(numeric_values) == 2:
        return ", ".join(_format_numeric_display(value) for value in numeric_values)
    return (
        f"{_format_numeric_display(numeric_values[0])}"
        f" to {_format_numeric_display(numeric_values[-1])}"
    )


def _point_sort_key(feature_type: str, raw_x_value: Any) -> Any:
    """Sort categorical points lexically and numeric points by decoded value."""
    decoded = (
        raw_x_value
        if feature_type == "categorical"
        else _decode_numeric_x_for_display(raw_x_value)
    )
    if feature_type == "categorical":
        return (0, str(decoded))
    if isinstance(decoded, (int, float)):
        return (0, float(decoded))
    return (1, str(decoded))


class DatabaseService:
    """Service for managing users and their shape function edits."""

    def get_db(self) -> Session:
        """Get a new database session."""
        return SessionLocal()

    @staticmethod
    def _serialize_user(user: User, *, is_new: bool = False) -> Dict[str, Any]:
        """Convert a user ORM instance into the API payload used by user endpoints."""
        return {
            "id": user.id,
            "name": user.name,
            "profession": user.profession,
            "created_at": user.created_at.isoformat(),
            "is_new": is_new,
            "is_superadmin": bool(user.is_superadmin),
        }

    @staticmethod
    def _submission_sort_key(submission: Dict[str, Any]) -> Tuple[datetime, int]:
        """Use the newest row in a submission as its ordering key."""
        return (
            submission.get("_latest_created_at") or datetime.min,
            submission.get("_latest_row_id") or 0,
        )

    def _get_effective_submissions(
        self,
        db: Session,
        *,
        user_id: Optional[int] = None,
        include_user_names: bool = False,
    ) -> List[Dict[str, Any]]:
        """Group rows by submission and keep only the newest one per user/feature."""
        query = db.query(ShapeFunctionEdit)
        if user_id is not None:
            query = query.filter(ShapeFunctionEdit.user_id == user_id)

        rows = (
            query.order_by(
                ShapeFunctionEdit.user_id,
                ShapeFunctionEdit.feature_name,
                ShapeFunctionEdit.created_at,
                ShapeFunctionEdit.id,
                ShapeFunctionEdit.x_value,
            ).all()
        )
        if not rows:
            return []

        user_metadata: Dict[int, Dict[str, Any]] = {}
        if include_user_names:
            user_ids = sorted({row.user_id for row in rows})
            users = db.query(User).filter(User.id.in_(user_ids)).all()
            user_metadata = {
                user.id: {
                    "name": user.name,
                    "profession": user.profession,
                }
                for user in users
            }

        submissions_by_group: Dict[str, Dict[str, Any]] = {}
        ordered_submissions: List[Dict[str, Any]] = []

        for row in rows:
            group_id = row.submission_id or f"legacy:{row.id}"
            created_at = row.created_at or datetime.min

            submission = submissions_by_group.get(group_id)
            if submission is None:
                submission = {
                    "submission_id": row.submission_id or f"legacy-{row.id}",
                    "persisted_submission_id": row.submission_id,
                    "legacy_edit_id": None if row.submission_id else row.id,
                    "feature_name": row.feature_name,
                    "feature_type": row.feature_type,
                    "user_id": row.user_id,
                    "user_name": user_metadata.get(row.user_id, {}).get("name", "Unknown"),
                    "profession": user_metadata.get(row.user_id, {}).get("profession"),
                    "message": row.message or "",
                    "sureness": round(float(row.weight) * 10),
                    "rows": [],
                    "_latest_created_at": created_at,
                    "_latest_row_id": row.id,
                }
                submissions_by_group[group_id] = submission
                ordered_submissions.append(submission)

            submission["rows"].append(row)
            row_sort_key = (created_at, row.id)
            if row_sort_key > self._submission_sort_key(submission):
                submission["_latest_created_at"] = created_at
                submission["_latest_row_id"] = row.id

        latest_by_user_feature: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for submission in ordered_submissions:
            latest_key = (submission["user_id"], submission["feature_name"])
            current = latest_by_user_feature.get(latest_key)
            if current is None or self._submission_sort_key(submission) > self._submission_sort_key(current):
                latest_by_user_feature[latest_key] = submission

        return list(latest_by_user_feature.values())

    @staticmethod
    def _rows_to_storage_feature(rows: List[ShapeFunctionEdit]) -> Dict[str, Any]:
        """Convert grouped rows into the raw storage shape expected by the ML service."""
        first_row = rows[0]
        sorted_rows = sorted(
            rows,
            key=lambda row: _point_sort_key(first_row.feature_type, row.x_value),
        )
        return {
            "feature_name": first_row.feature_name,
            "feature_type": first_row.feature_type,
            "edited_points": [
                {
                    "x_value": row.x_value,
                    "y_value": row.y_offset,
                    "weight": row.weight,
                }
                for row in sorted_rows
            ],
        }

    def _build_numeric_line_preview(
        self,
        feature_name: str,
        rows: List[ShapeFunctionEdit],
    ) -> Optional[Dict[str, Any]]:
        """Build original vs weighted-edited line data for numeric edit-log previews."""
        if not rows or rows[0].feature_type != "numeric" or not ml_service.is_trained:
            return None

        if not ml_service.original_shape_functions:
            try:
                ml_service.get_shape_functions()
            except Exception:
                return None

        original = ml_service.original_shape_functions.get(feature_name)
        if not original:
            return None

        x_values = list(original.get("x_values", []))
        original_y_values = [float(y) for y in original.get("y_values", [])]
        if len(x_values) != len(original_y_values):
            return None

        weighted_offsets = {
            row.x_value: float(row.y_offset) * float(row.weight)
            for row in rows
        }
        weighted_y_values = []
        for x_value, original_y in zip(x_values, original_y_values):
            offset = ml_service.get_offset_for_feature_value(
                feature_name=feature_name,
                value=x_value,
                offsets=weighted_offsets,
            )
            weighted_y_values.append(float(original_y) + float(offset))

        return {
            "x_values": [float(x) for x in x_values],
            "original_y_values": original_y_values,
            "weighted_y_values": weighted_y_values,
        }

    @staticmethod
    def _parse_optional_datetime(raw_value: Any) -> Optional[datetime]:
        """Parse an optional ISO datetime string into a naive UTC-compatible datetime."""
        if not raw_value:
            return None
        if isinstance(raw_value, datetime):
            return raw_value.replace(tzinfo=None) if raw_value.tzinfo else raw_value
        raw_str = str(raw_value).strip()
        if not raw_str:
            return None
        if raw_str.endswith("Z"):
            raw_str = f"{raw_str[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(raw_str)
        except ValueError:
            return None
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed

    @staticmethod
    def _get_or_create_import_user(
        db: Session,
        user_name: str,
        profession: Optional[str] = None,
    ) -> User:
        """Resolve an imported edit author to an existing or placeholder user."""
        existing_user = db.query(User).filter(User.name == user_name).first()
        if existing_user is not None:
            if profession and not existing_user.profession:
                existing_user.profession = profession
            return existing_user

        user = User(
            name=user_name,
            password_hash=None,
            profession=profession,
            is_superadmin=False,
        )
        db.add(user)
        db.flush()
        return user

    # ==================== User Operations ====================

    def get_or_create_user(self, name: str) -> Dict[str, Any]:
        """Get existing user or create a new one."""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.name == name).first()
            if user is None:
                user = User(name=name)
                db.add(user)
                db.commit()
                db.refresh(user)
                is_new = True
            else:
                is_new = False

            return self._serialize_user(user, is_new=is_new)
        finally:
            db.close()

    def get_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get user by name."""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.name == name).first()
            if user:
                return self._serialize_user(user)
            return None
        finally:
            db.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                return self._serialize_user(user)
            return None
        finally:
            db.close()

    def create_user_with_password(
        self,
        username: str,
        password: str,
        profession: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> Dict[str, Any]:
        """Create a new user with a password hash."""
        db = self.get_db()
        try:
            existing = db.query(User).filter(User.name == username).first()
            if existing:
                raise ValueError("User already exists")
            user = User(
                name=username,
                password_hash=hash_password(password),
                profession=profession,
                is_superadmin=is_superadmin,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return self._serialize_user(user)
        finally:
            db.close()

    def verify_user_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify username/password and return user info if valid."""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.name == username).first()
            if user is None or not user.password_hash:
                return None
            if not verify_password(password, user.password_hash):
                return None
            return self._serialize_user(user)
        finally:
            db.close()

    def ensure_superadmin(self) -> Optional[Dict[str, Any]]:
        """Ensure the superadmin user exists in the database."""
        if not settings.superadmin_password:
            return None
        db = self.get_db()
        try:
            user = db.query(User).filter(User.name == settings.superadmin_username).first()
            if user is None:
                user = User(
                    name=settings.superadmin_username,
                    password_hash=hash_password(settings.superadmin_password),
                    is_superadmin=True,
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            elif not user.is_superadmin:
                user.is_superadmin = True
                db.commit()
                db.refresh(user)
            return self._serialize_user(user)
        finally:
            db.close()

    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get stored preferences for a user (returns empty dict if none set)."""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                return {}
            return user.preferences or {}
        finally:
            db.close()

    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Merge and persist user preferences. Returns the updated preferences dict."""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                raise ValueError(f"User {user_id} not found")
            current = dict(user.preferences or {})
            current.update(preferences)
            user.preferences = current
            db.commit()
            db.refresh(user)
            return user.preferences or {}
        finally:
            db.close()

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users."""
        db = self.get_db()
        try:
            users = db.query(User).all()
            return [self._serialize_user(user) for user in users]
        finally:
            db.close()

    def get_users_with_edits(self) -> List[Dict[str, Any]]:
        """Get all users who have made at least one edit."""
        db = self.get_db()
        try:
            # Get distinct user IDs from edits
            user_ids_with_edits = db.query(ShapeFunctionEdit.user_id).distinct().all()
            user_ids = [uid[0] for uid in user_ids_with_edits]
            
            if not user_ids:
                return []
            
            users = db.query(User).filter(User.id.in_(user_ids)).all()
            return [self._serialize_user(user) for user in users]
        finally:
            db.close()

    # ==================== Invite Operations ====================

    def create_invite_token(self, created_by_user_id: Optional[int]) -> Dict[str, Any]:
        """Create a new invite token for registration."""
        db = self.get_db()
        try:
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(hours=settings.invite_token_ttl_hours)
            invite = InviteToken(
                token=token,
                created_by_user_id=created_by_user_id,
                expires_at=expires_at,
            )
            db.add(invite)
            db.commit()
            db.refresh(invite)
            return {
                "token": invite.token,
                "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
            }
        finally:
            db.close()

    def consume_invite_token(self, token: str) -> bool:
        """Mark an invite token as used if valid."""
        db = self.get_db()
        try:
            invite = db.query(InviteToken).filter(InviteToken.token == token).first()
            if invite is None:
                return False
            if invite.used_at is not None:
                return False
            if invite.expires_at and invite.expires_at < datetime.utcnow():
                return False
            invite.used_at = datetime.utcnow()
            db.commit()
            return True
        finally:
            db.close()

    # ==================== Edit Operations ====================

    def save_user_edits(self, user_id: int, edited_shape_functions: List[Dict[str, Any]]) -> bool:
        """Save or update shape function edits for a user.
        
        Only deletes/replaces edits for the specific features being submitted,
        preserving edits for other features.
        """
        db = self.get_db()
        try:
            # Get the list of feature names being submitted
            feature_names_to_update = [sf["feature_name"] for sf in edited_shape_functions]
            
            # Delete existing edits only for the features being updated
            if feature_names_to_update:
                db.query(ShapeFunctionEdit).filter(
                    ShapeFunctionEdit.user_id == user_id,
                    ShapeFunctionEdit.feature_name.in_(feature_names_to_update)
                ).delete(synchronize_session=False)
            
            # Insert new edits
            for sf in edited_shape_functions:
                feature_name = sf["feature_name"]
                feature_type = sf["feature_type"]
                submission_id = sf.get("submission_id")
                
                for point in sf.get("edited_points", []):
                    x_value = str(point["x_value"])
                    y_offset = float(point["y_value"])  # This is the offset/new value
                    weight = float(point.get("weight", 0.5))  # Default to 0.5 if not provided
                    message = str(point.get("message", ""))  # Commit message
                    
                    edit = ShapeFunctionEdit(
                        user_id=user_id,
                        feature_name=feature_name,
                        feature_type=feature_type,
                        x_value=x_value,
                        y_offset=y_offset,
                        weight=weight,
                        message=message,
                        submission_id=submission_id,
                    )
                    db.add(edit)
            
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def get_user_edits(self, user_id: int) -> Dict[str, Dict[str, float]]:
        """Get all edits for a specific user as a dictionary."""
        db = self.get_db()
        try:
            submissions = self._get_effective_submissions(db, user_id=user_id)
            result: Dict[str, Dict[str, float]] = {}
            for submission in submissions:
                feature_points = {}
                for edit in submission["rows"]:
                    feature_points[edit.x_value] = edit.y_offset
                result[submission["feature_name"]] = feature_points
            return result
        finally:
            db.close()

    def get_user_edits_raw(self, user_id: int) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get all edits for a user as {feature_name: {x_value_str: {offset, weight}}}."""
        db = self.get_db()
        try:
            result: Dict[str, Dict[str, Dict[str, float]]] = {}
            submissions = self._get_effective_submissions(db, user_id=user_id)
            for submission in submissions:
                feature_name = submission["feature_name"]
                result[feature_name] = {}
                for edit in submission["rows"]:
                    result[feature_name][edit.x_value] = {
                        "offset": edit.y_offset,
                        "weight": edit.weight,
                    }
            return result
        finally:
            db.close()

    def get_user_edits_as_list(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all edits for a specific user as a list of shape functions (display-oriented)."""
        db = self.get_db()
        try:
            submissions = self._get_effective_submissions(db, user_id=user_id)
            features = []
            for submission in sorted(submissions, key=lambda item: item["feature_name"]):
                storage_feature = self._rows_to_storage_feature(submission["rows"])
                display_points = []
                for point in storage_feature["edited_points"]:
                    display_points.append(
                        {
                            "x_value": (
                                point["x_value"]
                                if storage_feature["feature_type"] == "categorical"
                                else _decode_numeric_x_for_display(point["x_value"])
                            ),
                            "y_value": point["y_value"],
                            "weight": point["weight"],
                        }
                    )
                features.append(
                    {
                        "feature_name": storage_feature["feature_name"],
                        "feature_type": storage_feature["feature_type"],
                        "edited_points": display_points,
                    }
                )
            return features
        finally:
            db.close()

    def get_user_edits_storage_as_list(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all edits for a user preserving raw storage keys (for model loading)."""
        db = self.get_db()
        try:
            submissions = self._get_effective_submissions(db, user_id=user_id)
            return [
                self._rows_to_storage_feature(submission["rows"])
                for submission in sorted(submissions, key=lambda item: item["feature_name"])
            ]
        finally:
            db.close()

    def clear_user_edits(self, user_id: int) -> bool:
        """Clear all edits for a specific user."""
        db = self.get_db()
        try:
            db.query(ShapeFunctionEdit).filter(
                ShapeFunctionEdit.user_id == user_id
            ).delete()
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def clear_user_feature_edits(self, user_id: int, feature_name: str) -> bool:
        """Clear edits for a specific user and feature."""
        db = self.get_db()
        try:
            db.query(ShapeFunctionEdit).filter(
                ShapeFunctionEdit.user_id == user_id,
                ShapeFunctionEdit.feature_name == feature_name
            ).delete()
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def clear_all_shape_edits(self) -> bool:
        """Clear all persisted shape-function edits and deletion notifications."""
        db = self.get_db()
        try:
            db.query(ShapeFunctionEdit).delete()
            db.query(DeletedEditNotification).delete()
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def export_shape_function_edits_artifact(self) -> Dict[str, Any]:
        """Export the active saved shape-function edits in a model-artifact friendly shape."""
        db = self.get_db()
        try:
            submissions = self._get_effective_submissions(db, include_user_names=True)
            users: List[Dict[str, Any]] = []
            edits: List[Dict[str, Any]] = []
            seen_user_names = set()
            total_submissions = 0

            for submission in sorted(
                submissions,
                key=lambda item: (
                    item["user_name"].lower(),
                    item["feature_name"].lower(),
                    self._submission_sort_key(item),
                ),
            ):
                user_name = str(submission["user_name"]).strip()
                if not user_name:
                    continue

                if user_name not in seen_user_names:
                    users.append(
                        {
                            "name": user_name,
                            "profession": submission.get("profession"),
                            "is_superadmin": False,
                        }
                    )
                    edits.append(
                        {
                            "user_name": user_name,
                            "shape_functions": [],
                        }
                    )
                    seen_user_names.add(user_name)

                user_edit_group = next(
                    group for group in edits if group["user_name"] == user_name
                )
                rows = sorted(
                    submission["rows"],
                    key=lambda row: _point_sort_key(
                        submission["feature_type"],
                        row.x_value,
                    ),
                )
                edited_points = []
                for row in rows:
                    point_payload = {
                        "x_value": row.x_value,
                        "y_value": float(row.y_offset),
                        "weight": float(row.weight),
                        "message": row.message or "",
                    }
                    if row.created_at:
                        point_payload["created_at"] = row.created_at.isoformat()
                    if row.updated_at:
                        point_payload["updated_at"] = row.updated_at.isoformat()
                    edited_points.append(point_payload)

                created_candidates = [row.created_at for row in rows if row.created_at]
                updated_candidates = [
                    row.updated_at or row.created_at for row in rows if row.updated_at or row.created_at
                ]
                shape_payload = {
                    "feature_name": submission["feature_name"],
                    "feature_type": submission["feature_type"],
                    "submission_id": submission["persisted_submission_id"] or submission["submission_id"],
                    "message": submission["message"],
                    "sureness": submission["sureness"],
                    "edited_points": edited_points,
                }
                if created_candidates:
                    shape_payload["created_at"] = min(created_candidates).isoformat()
                if updated_candidates:
                    shape_payload["updated_at"] = max(updated_candidates).isoformat()
                user_edit_group["shape_functions"].append(shape_payload)
                total_submissions += 1

            return {
                "included": True,
                "users": users,
                "edits": edits,
                "user_count": len(users),
                "submission_count": total_submissions,
            }
        finally:
            db.close()

    def import_shape_function_edits_artifact(self, payload: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Import saved shape-function edits from an artifact payload."""
        if not payload or not payload.get("included"):
            self.clear_all_shape_edits()
            return {
                "imported_edit_user_count": 0,
                "imported_edit_submission_count": 0,
            }

        users_payload = payload.get("users")
        edits_payload = payload.get("edits")
        if not isinstance(users_payload, list) or not isinstance(edits_payload, list):
            raise ValueError("Imported shape-function edits payload must include users and edits arrays")

        declared_user_names = {
            str(user.get("name", "")).strip()
            for user in users_payload
            if isinstance(user, dict) and str(user.get("name", "")).strip()
        }
        declared_user_professions = {
            str(user.get("name", "")).strip(): (
                str(user.get("profession", "")).strip() or None
            )
            for user in users_payload
            if isinstance(user, dict) and str(user.get("name", "")).strip()
        }

        db = self.get_db()
        try:
            db.query(ShapeFunctionEdit).delete()
            db.query(DeletedEditNotification).delete()

            imported_user_names = set()
            imported_submission_count = 0

            for user_group in edits_payload:
                if not isinstance(user_group, dict):
                    raise ValueError("Imported shape-function edits must group edits by user")

                user_name = str(user_group.get("user_name", "")).strip()
                if not user_name:
                    raise ValueError("Imported shape-function edits contain a user group without user_name")
                if declared_user_names and user_name not in declared_user_names:
                    raise ValueError(f"Imported shape-function edits reference undeclared user '{user_name}'")

                shape_functions = user_group.get("shape_functions")
                if not isinstance(shape_functions, list):
                    raise ValueError(f"Imported shape-function edits for '{user_name}' must include shape_functions")

                user = self._get_or_create_import_user(
                    db,
                    user_name,
                    declared_user_professions.get(user_name),
                )
                inserted_any_for_user = False

                for shape_function in shape_functions:
                    if not isinstance(shape_function, dict):
                        raise ValueError("Imported shape-function edit entries must be objects")

                    feature_name = str(shape_function.get("feature_name", "")).strip()
                    feature_type = str(shape_function.get("feature_type", "")).strip()
                    if feature_name not in ml_service.original_shape_functions:
                        raise ValueError(
                            f"Imported shape-function edits reference unknown feature '{feature_name}'"
                        )
                    if feature_type not in {"numeric", "categorical"}:
                        raise ValueError(
                            f"Imported shape-function edits have invalid feature_type for '{feature_name}'"
                        )

                    edited_points = shape_function.get("edited_points")
                    if not isinstance(edited_points, list) or not edited_points:
                        continue

                    submission_id = str(shape_function.get("submission_id", "")).strip() or uuid4().hex
                    feature_created_at = self._parse_optional_datetime(
                        shape_function.get("created_at")
                    )
                    feature_updated_at = self._parse_optional_datetime(
                        shape_function.get("updated_at")
                    )
                    feature_message = str(shape_function.get("message", "") or "")

                    for point in edited_points:
                        if not isinstance(point, dict):
                            raise ValueError(
                                f"Imported shape-function edit points for '{feature_name}' must be objects"
                            )
                        x_value = str(point.get("x_value"))
                        y_value = float(point.get("y_value"))
                        weight = float(point.get("weight", 0.5))
                        message = str(point.get("message", feature_message) or "")
                        created_at = self._parse_optional_datetime(point.get("created_at")) or feature_created_at
                        updated_at = self._parse_optional_datetime(point.get("updated_at")) or feature_updated_at or created_at

                        edit = ShapeFunctionEdit(
                            user_id=user.id,
                            feature_name=feature_name,
                            feature_type=feature_type,
                            x_value=x_value,
                            y_offset=y_value,
                            weight=weight,
                            message=message,
                            submission_id=submission_id,
                            created_at=created_at or datetime.utcnow(),
                            updated_at=updated_at or created_at or datetime.utcnow(),
                        )
                        db.add(edit)

                    imported_submission_count += 1
                    inserted_any_for_user = True

                if inserted_any_for_user:
                    imported_user_names.add(user_name)

            db.commit()
            return {
                "imported_edit_user_count": len(imported_user_names),
                "imported_edit_submission_count": imported_submission_count,
            }
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    # ==================== Aggregation Operations ====================

    def get_combined_edits(self, weighted: bool = True) -> Dict[str, Dict[Any, float]]:
        """
        Get combined edits from all users.
        weighted=True  → each edit is multiplied by its confidence weight, then averaged.
        weighted=False → simple unweighted mean of raw offsets across all users.
        Returns format compatible with ml_service.shape_function_offsets:
        {feature_name: {x_value_or_index: avg_offset}}
        """
        db = self.get_db()
        try:
            submissions = self._get_effective_submissions(db)
            aggregate: Dict[str, Dict[Any, Dict[str, float]]] = {}

            for submission in submissions:
                feature_name = submission["feature_name"]
                feature_bucket = aggregate.setdefault(feature_name, {})
                for edit in submission["rows"]:
                    point_bucket = feature_bucket.setdefault(
                        edit.x_value,
                        {"offset_sum": 0.0, "user_count": 0.0},
                    )
                    contribution = (
                        float(edit.y_offset) * float(edit.weight)
                        if weighted
                        else float(edit.y_offset)
                    )
                    point_bucket["offset_sum"] += contribution
                    point_bucket["user_count"] += 1.0

            combined: Dict[str, Dict[Any, float]] = {}
            for feature_name, points in aggregate.items():
                combined[feature_name] = {}
                for x_value, stats in points.items():
                    user_count = stats["user_count"]
                    avg_offset = stats["offset_sum"] / user_count if user_count > 0 else 0.0
                    combined[feature_name][x_value] = avg_offset
            return combined
        finally:
            db.close()

    def get_combined_edits_detailed(self) -> Dict[str, Any]:
        """
        Get detailed combined edits from all users with statistics.
        Each edit is multiplied by its weight, then averaged across users.
        """
        db = self.get_db()
        try:
            submissions = self._get_effective_submissions(db)
            total_users = len({submission["user_id"] for submission in submissions})

            features: Dict[str, Dict[str, Any]] = {}
            point_stats: Dict[Tuple[str, Any], Dict[str, Any]] = {}

            for submission in submissions:
                feature_name = submission["feature_name"]
                feature_type = submission["feature_type"]
                features.setdefault(
                    feature_name,
                    {
                        "feature_name": feature_name,
                        "feature_type": feature_type,
                        "edited_points": [],
                    },
                )

                for edit in submission["rows"]:
                    point_key = (feature_name, edit.x_value)
                    weighted_value = float(edit.y_offset) * float(edit.weight)
                    stats = point_stats.setdefault(
                        point_key,
                        {
                            "feature_name": feature_name,
                            "feature_type": feature_type,
                            "x_value": edit.x_value,
                            "weighted_sum": 0.0,
                            "min_weighted": weighted_value,
                            "max_weighted": weighted_value,
                            "weight_sum": 0.0,
                            "user_count": 0,
                        },
                    )
                    stats["weighted_sum"] += weighted_value
                    stats["min_weighted"] = min(stats["min_weighted"], weighted_value)
                    stats["max_weighted"] = max(stats["max_weighted"], weighted_value)
                    stats["weight_sum"] += float(edit.weight)
                    stats["user_count"] += 1

            for stats in point_stats.values():
                feature = features[stats["feature_name"]]
                user_count = stats["user_count"]
                feature["edited_points"].append(
                    {
                        "x_value": (
                            stats["x_value"]
                            if stats["feature_type"] == "categorical"
                            else _decode_numeric_x_for_display(stats["x_value"])
                        ),
                        "y_value": (
                            stats["weighted_sum"] / user_count if user_count > 0 else 0.0
                        ),
                        "min_weighted": stats["min_weighted"],
                        "max_weighted": stats["max_weighted"],
                        "avg_weight": (
                            stats["weight_sum"] / user_count if user_count > 0 else 0.0
                        ),
                        "user_count": user_count,
                    }
                )

            feature_list = list(features.values())
            for feature in feature_list:
                feature["edited_points"].sort(
                    key=lambda point: _point_sort_key(
                        feature["feature_type"],
                        point["x_value"],
                    )
                )

            return {
                "total_users_with_edits": total_users,
                "shape_functions": sorted(
                    feature_list,
                    key=lambda feature: feature["feature_name"],
                ),
            }
        finally:
            db.close()

    def get_edit_logs(self) -> Dict[str, Any]:
        """
        Get detailed edit logs for all users, grouped by feature.
        Returns submissions grouped by feature, with nested point details.
        """
        db = self.get_db()
        try:
            submissions = self._get_effective_submissions(db, include_user_names=True)
            grouped_features: Dict[str, Dict[str, Any]] = {}

            for submission in submissions:
                feature_name = submission["feature_name"]
                feature_type = submission["feature_type"]
                rows = submission["rows"]

                grouped_features.setdefault(
                    feature_name,
                    {
                        "feature_name": feature_name,
                        "feature_type": feature_type,
                        "submissions": [],
                    },
                )

                points = []
                raw_input_total = 0.0
                weighted_total = 0.0
                for row in rows:
                    raw_input = float(row.y_offset)
                    weighted_result = float(row.y_offset) * float(row.weight)
                    raw_input_total += raw_input
                    weighted_total += weighted_result
                    points.append(
                        {
                            "edit_id": row.id,
                            "x_value": (
                                row.x_value
                                if row.feature_type == "categorical"
                                else _decode_numeric_x_for_display(row.x_value)
                            ),
                            "raw_input": raw_input,
                            "weighted_result": weighted_result,
                        }
                    )

                points.sort(
                    key=lambda point: _point_sort_key(
                        feature_type,
                        point["x_value"],
                    )
                )

                grouped_features[feature_name]["submissions"].append(
                    {
                        "submission_id": submission["submission_id"],
                        "persisted_submission_id": submission["persisted_submission_id"],
                        "legacy_edit_id": submission["legacy_edit_id"],
                        "feature_name": feature_name,
                        "feature_type": feature_type,
                        "user_id": submission["user_id"],
                        "user_name": submission["user_name"],
                        "profession": submission.get("profession"),
                        "created_at": submission["_latest_created_at"].isoformat(),
                        "sureness": submission["sureness"],
                        "message": submission["message"],
                        "point_count": len(rows),
                        "raw_input_total": raw_input_total,
                        "weighted_total": weighted_total,
                        "x_summary": _build_x_summary_from_rows(rows),
                        "points": points,
                        "line_preview": self._build_numeric_line_preview(
                            feature_name,
                            rows,
                        ),
                    }
                )

            feature_list = sorted(
                grouped_features.values(),
                key=lambda feature: feature["feature_name"],
            )
            for feature in feature_list:
                feature["submissions"].sort(
                    key=lambda submission: submission["created_at"],
                    reverse=True,
                )
            return {"features": feature_list}
        finally:
            db.close()

    # ==================== Edit Deletion Operations ====================

    def delete_edit(self, edit_id: int, deleted_by_user_id: int, reason: str) -> bool:
        """Delete a specific edit by ID. Creates a notification if the deleter is different from the edit owner."""
        db = self.get_db()
        try:
            edit = db.query(ShapeFunctionEdit).filter(ShapeFunctionEdit.id == edit_id).first()
            if edit is None:
                return False
            
            # If deleted by a different user, create a notification
            if edit.user_id != deleted_by_user_id:
                notification = DeletedEditNotification(
                    target_user_id=edit.user_id,
                    deleted_by_user_id=deleted_by_user_id,
                    feature_name=edit.feature_name,
                    x_value=edit.x_value,
                    reason=reason
                )
                db.add(notification)
            
            db.delete(edit)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def delete_submission(
        self,
        submission_id: str,
        deleted_by_user_id: int,
        reason: str,
    ) -> bool:
        """Delete all point rows belonging to a single submitted curve edit."""
        db = self.get_db()
        try:
            edits = (
                db.query(ShapeFunctionEdit)
                .filter(ShapeFunctionEdit.submission_id == submission_id)
                .order_by(ShapeFunctionEdit.created_at, ShapeFunctionEdit.x_value)
                .all()
            )
            if not edits:
                return False

            first_edit = edits[0]
            x_summary = _build_x_summary_from_rows(edits)

            if first_edit.user_id != deleted_by_user_id:
                notification = DeletedEditNotification(
                    target_user_id=first_edit.user_id,
                    deleted_by_user_id=deleted_by_user_id,
                    feature_name=first_edit.feature_name,
                    x_value=x_summary or str(first_edit.x_value),
                    submission_id=submission_id,
                    point_count=len(edits),
                    x_summary=x_summary or None,
                    reason=reason,
                )
                db.add(notification)

            for edit in edits:
                db.delete(edit)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    # ==================== Notification Operations ====================

    def get_unseen_notifications(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all unseen deletion notifications for a user."""
        db = self.get_db()
        try:
            notifications = db.query(DeletedEditNotification).filter(
                DeletedEditNotification.target_user_id == user_id,
                DeletedEditNotification.seen == False
            ).order_by(DeletedEditNotification.created_at.desc()).all()
            
            result = []
            for n in notifications:
                deleted_by_user = db.query(User).filter(User.id == n.deleted_by_user_id).first()
                result.append({
                    "id": n.id,
                    "feature_name": n.feature_name,
                    "x_value": n.x_value,
                    "submission_id": n.submission_id,
                    "point_count": n.point_count,
                    "x_summary": n.x_summary,
                    "reason": n.reason,
                    "deleted_by": deleted_by_user.name if deleted_by_user else "Unknown",
                    "created_at": n.created_at.isoformat()
                })
            
            return result
        finally:
            db.close()

    def mark_notifications_seen(self, user_id: int) -> bool:
        """Mark all notifications for a user as seen."""
        db = self.get_db()
        try:
            db.query(DeletedEditNotification).filter(
                DeletedEditNotification.target_user_id == user_id,
                DeletedEditNotification.seen == False
            ).update({"seen": True})
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    # ==================== Database Management ====================

    def reset_all_data(self) -> bool:
        """Reset the entire database."""
        try:
            db_reset()
            return True
        except Exception as e:
            raise e


# Create singleton instance
db_service = DatabaseService()
