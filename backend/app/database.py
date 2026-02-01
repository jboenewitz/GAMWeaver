"""Database configuration and models using SQLAlchemy."""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pathlib import Path

# Database file location
DATABASE_DIR = Path(__file__).parent.parent.resolve()
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/igann_app.db"

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class User(Base):
    """User model to store user information."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user
    user = relationship("User", back_populates="edits")


def init_db():
    """Initialize the database, creating all tables."""
    Base.metadata.create_all(bind=engine)


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
