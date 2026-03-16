"""Data processing utilities for bike rental prediction."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path

# Determine the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_FILE = PROJECT_ROOT / "bike.csv"


def scale_values(values, new_min, new_max):
    """Scale values to a new range."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        old_min, old_max = values.min(), values.max()
        return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    arr = np.array(values)
    old_min, old_max = arr.min(), arr.max()
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def load_and_preprocess_data(csv_path: str = None):
    """Load and preprocess the bike rental dataset."""
    if csv_path is None:
        csv_path = DATA_FILE
    
    # Ensure the file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Remap numeric features
    df["Time of Day"] = df["hr"].astype(int)  # Keep as integer for numeric treatment
    df["Windspeed"] = scale_values(df["windspeed"], 0, 67)
    df["Temperature"] = scale_values(df["temp"], -8, 39)
    df["Perceived Temperature"] = scale_values(df["atemp"], -16, 50)
    df["Humidity"] = scale_values(df["hum"], 0, 100)
    
    # Remap categorical features
    df["Season"] = df["season"].replace({1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"})
    df["Weathersituation"] = df["weathersit"].replace({
        1: "Clear",
        2: "Cloudy",
        3: "Light Rain",
        4: "Heavy Rain",
    })
    
    # Create Day Type variable
    df["Type of Day"] = np.where(
        (df["workingday"] == 1) & (df["holiday"] == 0),
        "Working Day",
        np.where((df["workingday"] == 0) & (df["holiday"] == 0), "Weekend", "Holiday"),
    )
    
    # Drop NaN in target
    df.dropna(subset=["cnt"], inplace=True)
    
    # Set correct nan and drop
    df.replace("-", np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Set X and y
    y = pd.DataFrame(df["cnt"])
    
    # Columns to drop
    feature_to_drop = [
        "dteday", "season", "yr", "mnth", "hr", "holiday", "weathersit",
        "temp", "atemp", "hum", "windspeed", "cnt", "instant", "workingday",
        "casual", "registered", "weekday", "Perceived Temperature", "Season",
    ]
    
    X = df.drop(columns=feature_to_drop, inplace=False)
    
    return X, y, df


def get_preprocessor(X):
    """Create and return the preprocessing pipeline."""
    cat_features = ["Weathersituation", "Type of Day"]
    num_features = [feature for feature in X.columns if feature not in cat_features]
    
    # Create transformers
    num_transformer = Pipeline([
        ("num_imputer", SimpleImputer(strategy="mean")),
    ])
    
    cat_transformer = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
    ])
    
    # Column transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    
    return column_transformer, cat_features, num_features


def preprocess_data(X, preprocessor=None):
    """Apply preprocessing to the data."""
    if preprocessor is None:
        preprocessor, _, _ = get_preprocessor(X)
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)
    
    # Convert categorical columns to object type
    X_transformed = X_transformed.astype({
        "Type of Day": "object",
        "Weathersituation": "object",
    })
    
    return X_transformed, preprocessor


def prepare_training_data(csv_path: str = None, test_size: float = 0.2, random_state: int = 42):
    """Load, preprocess and split the data for training."""
    X, y, df = load_and_preprocess_data(csv_path)
    X_processed, preprocessor = preprocess_data(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, df
