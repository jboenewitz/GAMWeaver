"""Database service layer for user and edit management."""

from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any, Optional
from .database import SessionLocal, User, ShapeFunctionEdit, reset_database as db_reset


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
                "is_new": is_new
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
                    "created_at": user.created_at.isoformat()
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
                    "created_at": user.created_at.isoformat()
                }
            return None
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
                    "created_at": user.created_at.isoformat()
                }
                for user in users
            ]
        finally:
            db.close()

    # ==================== Edit Operations ====================

    def save_user_edits(self, user_id: int, edited_shape_functions: List[Dict[str, Any]]) -> bool:
        """Save or update shape function edits for a user."""
        db = self.get_db()
        try:
            # Delete existing edits for this user
            db.query(ShapeFunctionEdit).filter(ShapeFunctionEdit.user_id == user_id).delete()
            
            # Insert new edits
            for sf in edited_shape_functions:
                feature_name = sf["feature_name"]
                feature_type = sf["feature_type"]
                
                for point in sf.get("edited_points", []):
                    x_value = str(point["x_value"])
                    y_offset = float(point["y_value"])  # This is the offset/new value
                    
                    edit = ShapeFunctionEdit(
                        user_id=user_id,
                        feature_name=feature_name,
                        feature_type=feature_type,
                        x_value=x_value,
                        y_offset=y_offset
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
                    "y_value": edit.y_offset
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

    # ==================== Aggregation Operations ====================

    def get_combined_edits(self) -> Dict[str, Dict[str, float]]:
        """
        Get combined edits from all users.
        For each feature/x_value combination, average the y_offsets from all users.
        """
        db = self.get_db()
        try:
            # Query to get average offset per feature/x_value
            results = db.query(
                ShapeFunctionEdit.feature_name,
                ShapeFunctionEdit.feature_type,
                ShapeFunctionEdit.x_value,
                func.avg(ShapeFunctionEdit.y_offset).label("avg_offset"),
                func.count(ShapeFunctionEdit.user_id).label("user_count")
            ).group_by(
                ShapeFunctionEdit.feature_name,
                ShapeFunctionEdit.x_value
            ).all()
            
            # Organize by feature name
            combined = {}
            for row in results:
                if row.feature_name not in combined:
                    combined[row.feature_name] = {}
                combined[row.feature_name][row.x_value] = row.avg_offset
            
            return combined
        finally:
            db.close()

    def get_combined_edits_detailed(self) -> Dict[str, Any]:
        """
        Get detailed combined edits from all users with statistics.
        """
        db = self.get_db()
        try:
            # Query to get statistics per feature/x_value
            results = db.query(
                ShapeFunctionEdit.feature_name,
                ShapeFunctionEdit.feature_type,
                ShapeFunctionEdit.x_value,
                func.avg(ShapeFunctionEdit.y_offset).label("avg_offset"),
                func.min(ShapeFunctionEdit.y_offset).label("min_offset"),
                func.max(ShapeFunctionEdit.y_offset).label("max_offset"),
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
                features[row.feature_name]["edited_points"].append({
                    "x_value": row.x_value if row.feature_type == "categorical" else float(row.x_value),
                    "y_value": row.avg_offset,
                    "min_offset": row.min_offset,
                    "max_offset": row.max_offset,
                    "user_count": row.user_count
                })
            
            return {
                "total_users_with_edits": total_users,
                "shape_functions": list(features.values())
            }
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
