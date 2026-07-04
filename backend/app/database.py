"""Database configuration and models using SQLAlchemy."""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from datetime import datetime

from .config import settings

# Database URL (PostgreSQL in production, SQLite fallback for development)
DATABASE_URL = settings.database_url

# Create engine
db_url = make_url(DATABASE_URL)
connect_args = {}
if db_url.drivername.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class User(Base):
    """User model to store user information."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(Text, nullable=True, default=None)
    is_superadmin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences = Column(JSON, nullable=True, default=None)  # User preferences (chart colors, etc.)
    
    # Relationship to shape function edits
    edits = relationship("ShapeFunctionEdit", back_populates="user", cascade="all, delete-orphan")


class ShapeFunctionEdit(Base):
    """Model to store individual shape function edits by users."""
    __tablename__ = "shape_function_edits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    feature_name = Column(String(100), nullable=False)
    feature_type = Column(String(20), nullable=False)  # 'numeric' or 'categorical'
    x_value = Column(String(100), nullable=False)  # Store as string to handle both numeric and categorical
    y_offset = Column(Float, nullable=False)  # The offset/change applied to the y value
    weight = Column(Float, nullable=False, default=0.5)  # Sureness weight (0.1 to 1.0, derived from 1-10 slider)
    message = Column(Text, nullable=True, default="")  # Commit message for the edit
    submission_id = Column(String(64), nullable=True, default=None)  # Groups point rows from a single curve submission
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user
    user = relationship("User", back_populates="edits")


class DeletedEditNotification(Base):
    """Model to store notifications for users whose edits were deleted."""
    __tablename__ = "deleted_edit_notifications"

    id = Column(Integer, primary_key=True, index=True)
    target_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # User who made the edit (receives notification)
    deleted_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # User who deleted the edit
    feature_name = Column(String(100), nullable=False)
    x_value = Column(String(100), nullable=False)
    submission_id = Column(String(64), nullable=True, default=None)
    point_count = Column(Integer, nullable=True, default=None)
    x_summary = Column(Text, nullable=True, default=None)
    reason = Column(Text, nullable=False)
    seen = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    target_user = relationship("User", foreign_keys=[target_user_id])
    deleted_by_user = relationship("User", foreign_keys=[deleted_by_user_id])


class InviteToken(Base):
    """Invite token for controlled registration."""
    __tablename__ = "invite_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(128), unique=True, index=True, nullable=False)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, default=None)
    used_at = Column(DateTime, nullable=True, default=None)

    created_by = relationship("User", foreign_keys=[created_by_user_id])


def init_db():
    """Initialize the database, creating all tables."""
    Base.metadata.create_all(bind=engine)
    # Migrate: add columns to existing tables if absent
    try:
        inspector = inspect(engine)
        user_columns = [col["name"] for col in inspector.get_columns("users")]
        if "password_hash" not in user_columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT NULL"))
                conn.commit()
        if "is_superadmin" not in user_columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN is_superadmin BOOLEAN DEFAULT 0"))
                conn.commit()
        if "preferences" not in user_columns:
            with engine.connect() as conn:
                if engine.dialect.name == "postgresql":
                    conn.execute(text("ALTER TABLE users ADD COLUMN preferences JSONB DEFAULT NULL"))
                else:
                    conn.execute(text("ALTER TABLE users ADD COLUMN preferences TEXT DEFAULT NULL"))
                conn.commit()

        edit_columns = [
            col["name"] for col in inspector.get_columns("shape_function_edits")
        ]
        if "submission_id" not in edit_columns:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE shape_function_edits ADD COLUMN submission_id TEXT DEFAULT NULL"
                    )
                )
                conn.commit()

        notification_columns = [
            col["name"]
            for col in inspector.get_columns("deleted_edit_notifications")
        ]
        if "submission_id" not in notification_columns:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE deleted_edit_notifications ADD COLUMN submission_id TEXT DEFAULT NULL"
                    )
                )
                conn.commit()
        if "point_count" not in notification_columns:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE deleted_edit_notifications ADD COLUMN point_count INTEGER DEFAULT NULL"
                    )
                )
                conn.commit()
        if "x_summary" not in notification_columns:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE deleted_edit_notifications ADD COLUMN x_summary TEXT DEFAULT NULL"
                    )
                )
                conn.commit()
    except Exception:
        pass  # Table may not exist yet on fresh installs — create_all handles it


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_database():
    """Reset the database by dropping and recreating all tables."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


# Initialize database on module import
init_db()
