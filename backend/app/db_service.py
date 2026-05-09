"""Database service layer for user and edit management."""

from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any, Optional
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


class DatabaseService:
    """Service for managing users and their shape function edits."""

    def get_db(self) -> Session:
        """Get a new database session."""
        return SessionLocal()

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
            
            return {
                "id": user.id,
                "name": user.name,
                "created_at": user.created_at.isoformat(),
                "is_new": is_new,
                "is_superadmin": bool(user.is_superadmin),
            }
        finally:
            db.close()

    def get_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get user by name."""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.name == name).first()
            if user:
                return {
                    "id": user.id,
                    "name": user.name,
                    "created_at": user.created_at.isoformat(),
                    "is_superadmin": bool(user.is_superadmin),
                }
            return None
        finally:
            db.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                return {
                    "id": user.id,
                    "name": user.name,
                    "created_at": user.created_at.isoformat(),
                    "is_superadmin": bool(user.is_superadmin),
                }
            return None
        finally:
            db.close()

    def create_user_with_password(self, username: str, password: str, is_superadmin: bool = False) -> Dict[str, Any]:
        """Create a new user with a password hash."""
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
            return {
                "id": user.id,
                "name": user.name,
                "created_at": user.created_at.isoformat(),
                "is_superadmin": bool(user.is_superadmin),
            }
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
            return {
                "id": user.id,
                "name": user.name,
                "created_at": user.created_at.isoformat(),
                "is_superadmin": bool(user.is_superadmin),
            }
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
            return {
                "id": user.id,
                "name": user.name,
                "created_at": user.created_at.isoformat(),
                "is_superadmin": bool(user.is_superadmin),
            }
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
            return [
                {
                    "id": user.id,
                    "name": user.name,
                    "created_at": user.created_at.isoformat(),
                    "is_superadmin": bool(user.is_superadmin),
                }
                for user in users
            ]
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
            return [
                {
                    "id": user.id,
                    "name": user.name,
                    "created_at": user.created_at.isoformat(),
                    "is_superadmin": bool(user.is_superadmin),
                }
                for user in users
            ]
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
                        message=message
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
            edits = db.query(ShapeFunctionEdit).filter(
                ShapeFunctionEdit.user_id == user_id
            ).all()
            
            # Organize by feature name
            result = {}
            for edit in edits:
                if edit.feature_name not in result:
                    result[edit.feature_name] = {}
                result[edit.feature_name][edit.x_value] = edit.y_offset
            
            return result
        finally:
            db.close()

    def get_user_edits_raw(self, user_id: int) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get all edits for a user as {feature_name: {x_value_str: {offset, weight}}}."""
        db = self.get_db()
        try:
            edits = db.query(ShapeFunctionEdit).filter(
                ShapeFunctionEdit.user_id == user_id
            ).all()
            result: Dict[str, Dict[str, Dict[str, float]]] = {}
            for edit in edits:
                if edit.feature_name not in result:
                    result[edit.feature_name] = {}
                result[edit.feature_name][edit.x_value] = {
                    "offset": edit.y_offset,
                    "weight": edit.weight,
                }
            return result
        finally:
            db.close()

    def get_user_edits_as_list(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all edits for a specific user as a list of shape functions."""
        db = self.get_db()
        try:
            edits = db.query(ShapeFunctionEdit).filter(
                ShapeFunctionEdit.user_id == user_id
            ).all()
            
            # Organize by feature name
            features = {}
            for edit in edits:
                if edit.feature_name not in features:
                    features[edit.feature_name] = {
                        "feature_name": edit.feature_name,
                        "feature_type": edit.feature_type,
                        "edited_points": []
                    }
                features[edit.feature_name]["edited_points"].append({
                    "x_value": edit.x_value if edit.feature_type == "categorical" else float(edit.x_value),
                    "y_value": edit.y_offset,
                    "weight": edit.weight
                })
            
            return list(features.values())
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
            if weighted:
                results = db.query(
                    ShapeFunctionEdit.feature_name,
                    ShapeFunctionEdit.feature_type,
                    ShapeFunctionEdit.x_value,
                    func.sum(ShapeFunctionEdit.y_offset * ShapeFunctionEdit.weight).label("offset_sum"),
                    func.count(ShapeFunctionEdit.user_id).label("user_count")
                ).group_by(
                    ShapeFunctionEdit.feature_name,
                    ShapeFunctionEdit.x_value
                ).all()
            else:
                results = db.query(
                    ShapeFunctionEdit.feature_name,
                    ShapeFunctionEdit.feature_type,
                    ShapeFunctionEdit.x_value,
                    func.sum(ShapeFunctionEdit.y_offset).label("offset_sum"),
                    func.count(ShapeFunctionEdit.user_id).label("user_count")
                ).group_by(
                    ShapeFunctionEdit.feature_name,
                    ShapeFunctionEdit.x_value
                ).all()

            combined = {}
            for row in results:
                if row.feature_name not in combined:
                    combined[row.feature_name] = {}

                avg_offset = float(row.offset_sum) / float(row.user_count) if row.user_count > 0 else 0.0

                if row.feature_type == "categorical":
                    combined[row.feature_name][row.x_value] = avg_offset
                else:
                    try:
                        idx = int(row.x_value)
                        combined[row.feature_name][idx] = avg_offset
                    except ValueError:
                        combined[row.feature_name][row.x_value] = avg_offset

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
            # Query to get statistics per feature/x_value
            # Each edit is multiplied by its weight, then we average by user count
            results = db.query(
                ShapeFunctionEdit.feature_name,
                ShapeFunctionEdit.feature_type,
                ShapeFunctionEdit.x_value,
                func.sum(ShapeFunctionEdit.y_offset * ShapeFunctionEdit.weight).label("weighted_sum"),
                func.min(ShapeFunctionEdit.y_offset * ShapeFunctionEdit.weight).label("min_weighted"),
                func.max(ShapeFunctionEdit.y_offset * ShapeFunctionEdit.weight).label("max_weighted"),
                func.avg(ShapeFunctionEdit.weight).label("avg_weight"),
                func.count(ShapeFunctionEdit.user_id.distinct()).label("user_count")
            ).group_by(
                ShapeFunctionEdit.feature_name,
                ShapeFunctionEdit.x_value
            ).all()
            
            # Get total number of users who made edits
            total_users = db.query(
                func.count(ShapeFunctionEdit.user_id.distinct())
            ).scalar() or 0
            
            # Organize by feature name
            features = {}
            for row in results:
                if row.feature_name not in features:
                    features[row.feature_name] = {
                        "feature_name": row.feature_name,
                        "feature_type": row.feature_type,
                        "edited_points": []
                    }
                
                # Calculate: sum of (offset * weight) / number of users
                avg_weighted_offset = float(row.weighted_sum) / float(row.user_count) if row.user_count > 0 else 0.0
                
                features[row.feature_name]["edited_points"].append({
                    "x_value": row.x_value if row.feature_type == "categorical" else float(row.x_value),
                    "y_value": avg_weighted_offset,
                    "min_weighted": row.min_weighted,
                    "max_weighted": row.max_weighted,
                    "avg_weight": row.avg_weight,
                    "user_count": row.user_count
                })
            
            return {
                "total_users_with_edits": total_users,
                "shape_functions": list(features.values())
            }
        finally:
            db.close()

    def get_edit_logs(self) -> Dict[str, Any]:
        """
        Get detailed edit logs for all users, grouped by feature.
        Returns individual edits with user name, sureness (1-10), raw input, and weighted result.
        """
        db = self.get_db()
        try:
            # Query all edits with user info
            edits = db.query(ShapeFunctionEdit, User.name).join(
                User, ShapeFunctionEdit.user_id == User.id
            ).order_by(
                ShapeFunctionEdit.feature_name,
                User.name,
                ShapeFunctionEdit.x_value
            ).all()
            
            # Organize by feature name
            features = {}
            for edit, user_name in edits:
                if edit.feature_name not in features:
                    features[edit.feature_name] = {
                        "feature_name": edit.feature_name,
                        "feature_type": edit.feature_type,
                        "edits": []
                    }
                
                # Calculate sureness (1-10) from weight (0.1-1.0)
                sureness = round(edit.weight * 10)
                # Raw input is the y_offset before weight is applied
                raw_input = edit.y_offset
                # Weighted result is the value after applying the weight
                weighted_result = edit.y_offset * edit.weight
                
                features[edit.feature_name]["edits"].append({
                    "edit_id": edit.id,
                    "user_id": edit.user_id,
                    "user_name": user_name,
                    "x_value": edit.x_value if edit.feature_type == "categorical" else float(edit.x_value),
                    "sureness": sureness,
                    "raw_input": raw_input,
                    "weighted_result": weighted_result,
                    "message": edit.message or ""
                })
            
            return {
                "features": list(features.values())
            }
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
