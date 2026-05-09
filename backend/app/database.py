"""Database configuration and models using SQLAlchemy."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from .config import settings


DATABASE_URL = settings.database_url

db_url = make_url(DATABASE_URL)
connect_args = {}
if db_url.drivername.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """User model to store user information."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(Text, nullable=True, default=None)
    is_superadmin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences = Column(JSON, nullable=True, default=None)

    edits = relationship("ShapeFunctionEdit", back_populates="user", cascade="all, delete-orphan")


class Dataset(Base):
    """Persisted uploaded dataset."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    display_name = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    target_column = Column(String(255), nullable=False)
    schema_json = Column(JSON, nullable=False)
    csv_data = Column(LargeBinary, nullable=False)
    uploaded_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    uploaded_by = relationship("User", foreign_keys=[uploaded_by_user_id])
    model_versions = relationship("ModelVersion", back_populates="dataset", cascade="all, delete-orphan")


class ModelVersion(Base):
    """Persisted trained model artifact tied to a dataset."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    training_params = Column(JSON, nullable=False)
    train_size = Column(Integer, nullable=False)
    test_size = Column(Integer, nullable=False)
    schema_snapshot = Column(JSON, nullable=False)
    metrics_json = Column(JSON, nullable=False)
    artifact_blob = Column(LargeBinary, nullable=False)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="model_versions")
    created_by = relationship("User", foreign_keys=[created_by_user_id])


class ShapeFunctionEdit(Base):
    """Model to store individual shape function edits by users."""

    __tablename__ = "shape_function_edits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True, index=True)
    feature_name = Column(String(100), nullable=False)
    feature_type = Column(String(20), nullable=False)
    x_value = Column(String(100), nullable=False)
    y_offset = Column(Float, nullable=False)
    weight = Column(Float, nullable=False, default=0.5)
    message = Column(Text, nullable=True, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="edits")


class DeletedEditNotification(Base):
    """Model to store notifications for users whose edits were deleted."""

    __tablename__ = "deleted_edit_notifications"

    id = Column(Integer, primary_key=True, index=True)
    target_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    deleted_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True, index=True)
    feature_name = Column(String(100), nullable=False)
    x_value = Column(String(100), nullable=False)
    reason = Column(Text, nullable=False)
    seen = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

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


def _ensure_column_exists(table_name: str, column_name: str, ddl: str) -> None:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return

    columns = [column["name"] for column in inspector.get_columns(table_name)]
    if column_name in columns:
        return

    with engine.connect() as conn:
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl}"))
        conn.commit()


def init_db():
    """Initialize the database, creating all tables and additive migrations."""
    Base.metadata.create_all(bind=engine)

    try:
        _ensure_column_exists("users", "password_hash", "password_hash TEXT DEFAULT NULL")
        _ensure_column_exists("users", "is_superadmin", "is_superadmin BOOLEAN DEFAULT 0")
        if engine.dialect.name == "postgresql":
            _ensure_column_exists("users", "preferences", "preferences JSONB DEFAULT NULL")
        else:
            _ensure_column_exists("users", "preferences", "preferences JSON DEFAULT NULL")

        _ensure_column_exists(
            "shape_function_edits",
            "model_version_id",
            "model_version_id INTEGER DEFAULT NULL",
        )
        _ensure_column_exists(
            "deleted_edit_notifications",
            "model_version_id",
            "model_version_id INTEGER DEFAULT NULL",
        )
    except Exception:
        pass


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


init_db()
