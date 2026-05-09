"""Database service layer for user, dataset, and edit management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import secrets

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from .config import settings
from .database import (
    Dataset,
    DeletedEditNotification,
    InviteToken,
    ModelVersion,
    SessionLocal,
    ShapeFunctionEdit,
    User,
    reset_database as db_reset,
)
from .security import hash_password, verify_password


class DatabaseService:
    """Service for managing users, datasets, model versions, and edits."""

    def get_db(self) -> Session:
        return SessionLocal()

    def _user_to_dict(self, user: User, *, is_new: bool = False) -> Dict[str, Any]:
        return {
            "id": user.id,
            "name": user.name,
            "created_at": user.created_at.isoformat(),
            "is_new": is_new,
            "is_superadmin": bool(user.is_superadmin),
        }

    def _dataset_to_dict(self, dataset: Dataset, *, include_schema: bool = False) -> Dict[str, Any]:
        payload = {
            "id": dataset.id,
            "display_name": dataset.display_name,
            "original_filename": dataset.original_filename,
            "target_column": dataset.target_column,
            "created_at": dataset.created_at.isoformat(),
            "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else dataset.created_at.isoformat(),
            "uploaded_by_user_id": dataset.uploaded_by_user_id,
        }
        if include_schema:
            payload["schema"] = dataset.schema_json
        return payload

    def _model_version_to_dict(self, model_version: ModelVersion) -> Dict[str, Any]:
        return {
            "id": model_version.id,
            "dataset_id": model_version.dataset_id,
            "version_number": model_version.version_number,
            "training_params": model_version.training_params,
            "train_size": model_version.train_size,
            "test_size": model_version.test_size,
            "schema_snapshot": model_version.schema_snapshot,
            "metrics": model_version.metrics_json,
            "created_by_user_id": model_version.created_by_user_id,
            "created_at": model_version.created_at.isoformat(),
        }

    # ==================== User Operations ====================

    def get_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        db = self.get_db()
        try:
            user = db.query(User).filter(User.name == name).first()
            return self._user_to_dict(user) if user else None
        finally:
            db.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return self._user_to_dict(user) if user else None
        finally:
            db.close()

    def create_user_with_password(
        self,
        username: str,
        password: str,
        is_superadmin: bool = False,
    ) -> Dict[str, Any]:
        db = self.get_db()
        try:
            existing = db.query(User).filter(User.name == username).first()
            if existing:
                raise ValueError("User already exists")

            user = User(
                name=username,
                password_hash=hash_password(password),
                is_superadmin=is_superadmin,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return self._user_to_dict(user)
        finally:
            db.close()

    def verify_user_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        db = self.get_db()
        try:
            user = db.query(User).filter(User.name == username).first()
            if user is None or not user.password_hash:
                return None
            if not verify_password(password, user.password_hash):
                return None
            return self._user_to_dict(user)
        finally:
            db.close()

    def ensure_superadmin(self) -> Optional[Dict[str, Any]]:
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

            return self._user_to_dict(user)
        finally:
            db.close()

    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                return {}
            return user.preferences or {}
        finally:
            db.close()

    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> Dict[str, Any]:
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
        db = self.get_db()
        try:
            users = db.query(User).order_by(User.created_at.asc()).all()
            return [self._user_to_dict(user) for user in users]
        finally:
            db.close()

    # ==================== Dataset Operations ====================

    def create_dataset(
        self,
        display_name: str,
        original_filename: str,
        target_column: str,
        schema_json: Dict[str, Any],
        csv_data: bytes,
        uploaded_by_user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        db = self.get_db()
        try:
            dataset = Dataset(
                display_name=display_name,
                original_filename=original_filename,
                target_column=target_column,
                schema_json=schema_json,
                csv_data=csv_data,
                uploaded_by_user_id=uploaded_by_user_id,
            )
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            return self._dataset_to_dict(dataset, include_schema=True)
        finally:
            db.close()

    def list_datasets(self) -> List[Dict[str, Any]]:
        db = self.get_db()
        try:
            datasets = db.query(Dataset).order_by(Dataset.created_at.asc()).all()
            dataset_ids = [dataset.id for dataset in datasets]
            latest_by_dataset: Dict[int, ModelVersion] = {}

            if dataset_ids:
                versions = (
                    db.query(ModelVersion)
                    .filter(ModelVersion.dataset_id.in_(dataset_ids))
                    .order_by(ModelVersion.dataset_id.asc(), ModelVersion.version_number.desc())
                    .all()
                )
                for version in versions:
                    latest_by_dataset.setdefault(version.dataset_id, version)

            payload: List[Dict[str, Any]] = []
            for dataset in datasets:
                item = self._dataset_to_dict(dataset)
                latest = latest_by_dataset.get(dataset.id)
                item["latest_model_version_id"] = latest.id if latest else None
                item["latest_model_version_number"] = latest.version_number if latest else None
                payload.append(item)
            return payload
        finally:
            db.close()

    def get_dataset_by_id(
        self,
        dataset_id: int,
        *,
        include_schema: bool = True,
        include_csv_data: bool = False,
    ) -> Optional[Dict[str, Any]]:
        db = self.get_db()
        try:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset is None:
                return None

            payload = self._dataset_to_dict(dataset, include_schema=include_schema)
            if include_csv_data:
                payload["csv_data"] = dataset.csv_data
            return payload
        finally:
            db.close()

    def get_latest_model_version_for_dataset(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        db = self.get_db()
        try:
            version = (
                db.query(ModelVersion)
                .filter(ModelVersion.dataset_id == dataset_id)
                .order_by(ModelVersion.version_number.desc())
                .first()
            )
            return self._model_version_to_dict(version) if version else None
        finally:
            db.close()

    def get_model_version_by_id(self, model_version_id: int, *, include_artifact: bool = False) -> Optional[Dict[str, Any]]:
        db = self.get_db()
        try:
            version = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
            if version is None:
                return None

            payload = self._model_version_to_dict(version)
            if include_artifact:
                payload["artifact_blob"] = version.artifact_blob
            return payload
        finally:
            db.close()

    def create_model_version(
        self,
        dataset_id: int,
        training_params: Dict[str, Any],
        train_size: int,
        test_size: int,
        schema_snapshot: Dict[str, Any],
        metrics_json: Dict[str, Any],
        artifact_blob: bytes,
        created_by_user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        db = self.get_db()
        try:
            current_max = (
                db.query(func.max(ModelVersion.version_number))
                .filter(ModelVersion.dataset_id == dataset_id)
                .scalar()
            ) or 0

            version = ModelVersion(
                dataset_id=dataset_id,
                version_number=int(current_max) + 1,
                training_params=training_params,
                train_size=train_size,
                test_size=test_size,
                schema_snapshot=schema_snapshot,
                metrics_json=metrics_json,
                artifact_blob=artifact_blob,
                created_by_user_id=created_by_user_id,
            )
            db.add(version)
            db.commit()
            db.refresh(version)
            return self._model_version_to_dict(version)
        finally:
            db.close()

    def get_active_dataset_context(self, user_id: int) -> Dict[str, Any]:
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                raise ValueError(f"User {user_id} not found")

            datasets = db.query(Dataset).order_by(Dataset.created_at.asc()).all()
            datasets_payload = [self._dataset_to_dict(dataset) for dataset in datasets]
            if not datasets:
                return {
                    "datasets": datasets_payload,
                    "active_dataset": None,
                    "active_model_version": None,
                }

            prefs = dict(user.preferences or {})
            active_dataset_id = prefs.get("active_dataset_id")
            active_model_version_id = prefs.get("active_model_version_id")

            dataset = None
            if active_dataset_id is not None:
                dataset = db.query(Dataset).filter(Dataset.id == active_dataset_id).first()
            if dataset is None:
                dataset = datasets[0]
                active_dataset_id = dataset.id

            version = None
            if active_model_version_id is not None:
                version = db.query(ModelVersion).filter(ModelVersion.id == active_model_version_id).first()
                if version is not None and version.dataset_id != dataset.id:
                    version = None

            if version is None:
                version = (
                    db.query(ModelVersion)
                    .filter(ModelVersion.dataset_id == dataset.id)
                    .order_by(ModelVersion.version_number.desc())
                    .first()
                )
                active_model_version_id = version.id if version else None

            desired_prefs = {
                "active_dataset_id": active_dataset_id,
                "active_model_version_id": active_model_version_id,
            }
            if prefs.get("active_dataset_id") != desired_prefs["active_dataset_id"] or prefs.get(
                "active_model_version_id"
            ) != desired_prefs["active_model_version_id"]:
                prefs.update(desired_prefs)
                user.preferences = prefs
                db.commit()

            return {
                "datasets": datasets_payload,
                "active_dataset": self._dataset_to_dict(dataset, include_schema=True),
                "active_model_version": self._model_version_to_dict(version) if version else None,
            }
        finally:
            db.close()

    def set_active_dataset(self, user_id: int, dataset_id: int) -> Dict[str, Any]:
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                raise ValueError(f"User {user_id} not found")

            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset is None:
                raise ValueError(f"Dataset {dataset_id} not found")

            latest_version = (
                db.query(ModelVersion)
                .filter(ModelVersion.dataset_id == dataset_id)
                .order_by(ModelVersion.version_number.desc())
                .first()
            )

            prefs = dict(user.preferences or {})
            prefs["active_dataset_id"] = dataset_id
            prefs["active_model_version_id"] = latest_version.id if latest_version else None
            user.preferences = prefs
            db.commit()
            return {
                "active_dataset_id": dataset_id,
                "active_model_version_id": latest_version.id if latest_version else None,
            }
        finally:
            db.close()

    def set_active_model_version(self, user_id: int, dataset_id: int, model_version_id: int) -> Dict[str, Any]:
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                raise ValueError(f"User {user_id} not found")

            version = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
            if version is None or version.dataset_id != dataset_id:
                raise ValueError(f"Model version {model_version_id} not found for dataset {dataset_id}")

            prefs = dict(user.preferences or {})
            prefs["active_dataset_id"] = dataset_id
            prefs["active_model_version_id"] = model_version_id
            user.preferences = prefs
            db.commit()
            return {
                "active_dataset_id": dataset_id,
                "active_model_version_id": model_version_id,
            }
        finally:
            db.close()

    # ==================== Invite Operations ====================

    def create_invite_token(self, created_by_user_id: Optional[int]) -> Dict[str, Any]:
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

    def save_user_edits(
        self,
        user_id: int,
        model_version_id: int,
        edited_shape_functions: List[Dict[str, Any]],
    ) -> bool:
        db = self.get_db()
        try:
            feature_names_to_update = [sf["feature_name"] for sf in edited_shape_functions]
            if feature_names_to_update:
                (
                    db.query(ShapeFunctionEdit)
                    .filter(
                        ShapeFunctionEdit.user_id == user_id,
                        ShapeFunctionEdit.model_version_id == model_version_id,
                        ShapeFunctionEdit.feature_name.in_(feature_names_to_update),
                    )
                    .delete(synchronize_session=False)
                )

            for sf in edited_shape_functions:
                for point in sf.get("edited_points", []):
                    edit = ShapeFunctionEdit(
                        user_id=user_id,
                        model_version_id=model_version_id,
                        feature_name=sf["feature_name"],
                        feature_type=sf["feature_type"],
                        x_value=str(point["x_value"]),
                        y_offset=float(point["y_value"]),
                        weight=float(point.get("weight", 0.5)),
                        message=str(point.get("message", "")),
                    )
                    db.add(edit)

            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_user_edits_raw(
        self,
        user_id: int,
        model_version_id: int,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        db = self.get_db()
        try:
            edits = (
                db.query(ShapeFunctionEdit)
                .filter(
                    ShapeFunctionEdit.user_id == user_id,
                    ShapeFunctionEdit.model_version_id == model_version_id,
                )
                .all()
            )

            result: Dict[str, Dict[str, Dict[str, float]]] = {}
            for edit in edits:
                result.setdefault(edit.feature_name, {})
                result[edit.feature_name][edit.x_value] = {
                    "offset": edit.y_offset,
                    "weight": edit.weight,
                }
            return result
        finally:
            db.close()

    def get_user_edits_as_list(self, user_id: int, model_version_id: int) -> List[Dict[str, Any]]:
        db = self.get_db()
        try:
            edits = (
                db.query(ShapeFunctionEdit)
                .filter(
                    ShapeFunctionEdit.user_id == user_id,
                    ShapeFunctionEdit.model_version_id == model_version_id,
                )
                .all()
            )

            features: Dict[str, Dict[str, Any]] = {}
            for edit in edits:
                features.setdefault(
                    edit.feature_name,
                    {
                        "feature_name": edit.feature_name,
                        "feature_type": edit.feature_type,
                        "edited_points": [],
                    },
                )
                features[edit.feature_name]["edited_points"].append(
                    {
                        "x_value": edit.x_value if edit.feature_type == "categorical" else float(edit.x_value),
                        "y_value": edit.y_offset,
                        "weight": edit.weight,
                        "message": edit.message or "",
                    }
                )

            return list(features.values())
        finally:
            db.close()

    def clear_user_edits(self, user_id: int, model_version_id: int) -> bool:
        db = self.get_db()
        try:
            (
                db.query(ShapeFunctionEdit)
                .filter(
                    ShapeFunctionEdit.user_id == user_id,
                    ShapeFunctionEdit.model_version_id == model_version_id,
                )
                .delete()
            )
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def clear_user_feature_edits(self, user_id: int, model_version_id: int, feature_name: str) -> bool:
        db = self.get_db()
        try:
            (
                db.query(ShapeFunctionEdit)
                .filter(
                    ShapeFunctionEdit.user_id == user_id,
                    ShapeFunctionEdit.model_version_id == model_version_id,
                    ShapeFunctionEdit.feature_name == feature_name,
                )
                .delete()
            )
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_users_with_edits(self, model_version_id: Optional[int] = None) -> List[Dict[str, Any]]:
        db = self.get_db()
        try:
            query = db.query(ShapeFunctionEdit.user_id).distinct()
            if model_version_id is not None:
                query = query.filter(ShapeFunctionEdit.model_version_id == model_version_id)
            user_ids = [row[0] for row in query.all()]
            if not user_ids:
                return []

            users = db.query(User).filter(User.id.in_(user_ids)).order_by(User.name.asc()).all()
            return [self._user_to_dict(user) for user in users]
        finally:
            db.close()

    # ==================== Aggregation Operations ====================

    def get_combined_edits(self, model_version_id: int, weighted: bool = True) -> Dict[str, Dict[Any, float]]:
        db = self.get_db()
        try:
            if weighted:
                results = (
                    db.query(
                        ShapeFunctionEdit.feature_name,
                        ShapeFunctionEdit.feature_type,
                        ShapeFunctionEdit.x_value,
                        func.sum(ShapeFunctionEdit.y_offset * ShapeFunctionEdit.weight).label("offset_sum"),
                        func.count(ShapeFunctionEdit.user_id).label("user_count"),
                    )
                    .filter(ShapeFunctionEdit.model_version_id == model_version_id)
                    .group_by(ShapeFunctionEdit.feature_name, ShapeFunctionEdit.x_value)
                    .all()
                )
            else:
                results = (
                    db.query(
                        ShapeFunctionEdit.feature_name,
                        ShapeFunctionEdit.feature_type,
                        ShapeFunctionEdit.x_value,
                        func.sum(ShapeFunctionEdit.y_offset).label("offset_sum"),
                        func.count(ShapeFunctionEdit.user_id).label("user_count"),
                    )
                    .filter(ShapeFunctionEdit.model_version_id == model_version_id)
                    .group_by(ShapeFunctionEdit.feature_name, ShapeFunctionEdit.x_value)
                    .all()
                )

            combined: Dict[str, Dict[Any, float]] = {}
            for row in results:
                combined.setdefault(row.feature_name, {})
                avg_offset = float(row.offset_sum) / float(row.user_count) if row.user_count > 0 else 0.0
                if row.feature_type == "categorical":
                    combined[row.feature_name][row.x_value] = avg_offset
                else:
                    try:
                        combined[row.feature_name][int(row.x_value)] = avg_offset
                    except ValueError:
                        combined[row.feature_name][row.x_value] = avg_offset
            return combined
        finally:
            db.close()

    def get_combined_edits_detailed(self, model_version_id: int) -> Dict[str, Any]:
        db = self.get_db()
        try:
            results = (
                db.query(
                    ShapeFunctionEdit.feature_name,
                    ShapeFunctionEdit.feature_type,
                    ShapeFunctionEdit.x_value,
                    func.sum(ShapeFunctionEdit.y_offset * ShapeFunctionEdit.weight).label("weighted_sum"),
                    func.min(ShapeFunctionEdit.y_offset * ShapeFunctionEdit.weight).label("min_weighted"),
                    func.max(ShapeFunctionEdit.y_offset * ShapeFunctionEdit.weight).label("max_weighted"),
                    func.avg(ShapeFunctionEdit.weight).label("avg_weight"),
                    func.count(ShapeFunctionEdit.user_id.distinct()).label("user_count"),
                )
                .filter(ShapeFunctionEdit.model_version_id == model_version_id)
                .group_by(ShapeFunctionEdit.feature_name, ShapeFunctionEdit.x_value)
                .all()
            )

            total_users = (
                db.query(func.count(ShapeFunctionEdit.user_id.distinct()))
                .filter(ShapeFunctionEdit.model_version_id == model_version_id)
                .scalar()
            ) or 0

            features: Dict[str, Dict[str, Any]] = {}
            for row in results:
                features.setdefault(
                    row.feature_name,
                    {
                        "feature_name": row.feature_name,
                        "feature_type": row.feature_type,
                        "edited_points": [],
                    },
                )
                avg_weighted_offset = float(row.weighted_sum) / float(row.user_count) if row.user_count > 0 else 0.0
                features[row.feature_name]["edited_points"].append(
                    {
                        "x_value": row.x_value if row.feature_type == "categorical" else float(row.x_value),
                        "y_value": avg_weighted_offset,
                        "min_weighted": row.min_weighted,
                        "max_weighted": row.max_weighted,
                        "avg_weight": row.avg_weight,
                        "user_count": row.user_count,
                    }
                )

            return {
                "total_users_with_edits": total_users,
                "shape_functions": list(features.values()),
            }
        finally:
            db.close()

    def get_edit_logs(self, model_version_id: int) -> Dict[str, Any]:
        db = self.get_db()
        try:
            edits = (
                db.query(ShapeFunctionEdit, User.name)
                .join(User, ShapeFunctionEdit.user_id == User.id)
                .filter(ShapeFunctionEdit.model_version_id == model_version_id)
                .order_by(ShapeFunctionEdit.feature_name, User.name, ShapeFunctionEdit.x_value)
                .all()
            )

            features: Dict[str, Dict[str, Any]] = {}
            for edit, user_name in edits:
                features.setdefault(
                    edit.feature_name,
                    {
                        "feature_name": edit.feature_name,
                        "feature_type": edit.feature_type,
                        "edits": [],
                    },
                )
                sureness = round(edit.weight * 10)
                features[edit.feature_name]["edits"].append(
                    {
                        "edit_id": edit.id,
                        "user_id": edit.user_id,
                        "user_name": user_name,
                        "x_value": edit.x_value if edit.feature_type == "categorical" else float(edit.x_value),
                        "sureness": sureness,
                        "raw_input": edit.y_offset,
                        "weighted_result": edit.y_offset * edit.weight,
                        "message": edit.message or "",
                    }
                )

            return {"features": list(features.values())}
        finally:
            db.close()

    # ==================== Edit Deletion Operations ====================

    def delete_edit(self, edit_id: int, deleted_by_user_id: int, reason: str) -> bool:
        db = self.get_db()
        try:
            edit = db.query(ShapeFunctionEdit).filter(ShapeFunctionEdit.id == edit_id).first()
            if edit is None:
                return False

            if edit.user_id != deleted_by_user_id:
                notification = DeletedEditNotification(
                    target_user_id=edit.user_id,
                    deleted_by_user_id=deleted_by_user_id,
                    model_version_id=edit.model_version_id,
                    feature_name=edit.feature_name,
                    x_value=edit.x_value,
                    reason=reason,
                )
                db.add(notification)

            db.delete(edit)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ==================== Notification Operations ====================

    def get_unseen_notifications(
        self,
        user_id: int,
        model_version_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        db = self.get_db()
        try:
            query = (
                db.query(DeletedEditNotification)
                .filter(
                    DeletedEditNotification.target_user_id == user_id,
                    DeletedEditNotification.seen == False,
                )
                .order_by(desc(DeletedEditNotification.created_at))
            )
            if model_version_id is not None:
                query = query.filter(DeletedEditNotification.model_version_id == model_version_id)

            notifications = query.all()
            result: List[Dict[str, Any]] = []
            for notification in notifications:
                deleted_by_user = db.query(User).filter(User.id == notification.deleted_by_user_id).first()
                result.append(
                    {
                        "id": notification.id,
                        "feature_name": notification.feature_name,
                        "x_value": notification.x_value,
                        "reason": notification.reason,
                        "deleted_by": deleted_by_user.name if deleted_by_user else "Unknown",
                        "created_at": notification.created_at.isoformat(),
                        "model_version_id": notification.model_version_id,
                    }
                )
            return result
        finally:
            db.close()

    def mark_notifications_seen(self, user_id: int, model_version_id: Optional[int] = None) -> bool:
        db = self.get_db()
        try:
            query = db.query(DeletedEditNotification).filter(
                DeletedEditNotification.target_user_id == user_id,
                DeletedEditNotification.seen == False,
            )
            if model_version_id is not None:
                query = query.filter(DeletedEditNotification.model_version_id == model_version_id)
            query.update({"seen": True})
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ==================== Legacy Migration Helpers ====================

    def get_unscoped_edit_counts(self) -> Tuple[int, int]:
        db = self.get_db()
        try:
            edit_count = (
                db.query(func.count(ShapeFunctionEdit.id))
                .filter(ShapeFunctionEdit.model_version_id.is_(None))
                .scalar()
            ) or 0
            notification_count = (
                db.query(func.count(DeletedEditNotification.id))
                .filter(DeletedEditNotification.model_version_id.is_(None))
                .scalar()
            ) or 0
            return int(edit_count), int(notification_count)
        finally:
            db.close()

    def assign_legacy_model_version(self, model_version_id: int) -> None:
        db = self.get_db()
        try:
            (
                db.query(ShapeFunctionEdit)
                .filter(ShapeFunctionEdit.model_version_id.is_(None))
                .update({"model_version_id": model_version_id})
            )
            (
                db.query(DeletedEditNotification)
                .filter(DeletedEditNotification.model_version_id.is_(None))
                .update({"model_version_id": model_version_id})
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ==================== Database Management ====================

    def reset_all_data(self) -> bool:
        db_reset()
        return True


db_service = DatabaseService()
