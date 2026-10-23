"""
train_models.py
-----------------
This script trains the two Machine Learning models used by the project:

1. RandomForestRegressor  -> predicts the numeric Readiness Score (0-100)
2. RandomForestClassifier -> predicts the Readiness Status category
                              (Ready / Almost Ready / Needs More Preparation)

It reads the dataset from data/student_data.csv (created by
generate_dataset.py), splits it into training and testing sets, trains
both models, evaluates them, and saves:

- The trained models (models/regressor.pkl, models/classifier.pkl)
- The label encoder used for Revision_Status (models/revision_encoder.pkl)
- The label encoder used for Readiness_Status (models/status_encoder.pkl)
- A metrics summary (models/metrics.pkl) that app.py reads for the
  "Model Performance" page.

Run this AFTER generate_dataset.py:
    python train_models.py
"""

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
)

DATA_PATH = os.path.join("data", "student_data.csv")
MODELS_DIR = "models"

# Feature columns used to train both models
FEATURE_COLUMNS = [
    "Study_Hours_Per_Day",
    "Total_Chapters",
    "Chapters_Completed",
    "Mock_Test_Score",
    "Attendance_Percentage",
    "Days_Left_For_Exam",
    "Confidence_Level",
    "Revision_Status_Encoded",
]


def load_and_prepare_data():
    """
    Loads the dataset and encodes the categorical 'Revision_Status'
    column into numbers, since Machine Learning models require
    numeric input.
    """
    df = pd.read_csv(DATA_PATH)

    # Encode Revision_Status (Not Started / In Progress / Completed) -> 0, 1, 2
    revision_encoder = LabelEncoder()
    df["Revision_Status_Encoded"] = revision_encoder.fit_transform(df["Revision_Status"])

    # Encode Readiness_Status (target for classification) -> numeric labels
    status_encoder = LabelEncoder()
    df["Readiness_Status_Encoded"] = status_encoder.fit_transform(df["Readiness_Status"])

    return df, revision_encoder, status_encoder


def train_regression_model(df):
    """
    Trains a RandomForestRegressor to predict the numeric Readiness_Score.
    Returns the trained model and evaluation metrics.
    """
    X = df[FEATURE_COLUMNS]
    y = df["Readiness_Score"]

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "mae": round(mae, 3),
        "r2_score": round(r2, 3),
    }

    return model, metrics


def train_classification_model(df):
    """
    Trains a RandomForestClassifier to predict the Readiness_Status category.
    Returns the trained model and evaluation metrics.
    """
    X = df[FEATURE_COLUMNS]
    y = df["Readiness_Status_Encoded"]

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the model
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    metrics = {
        "accuracy": round(accuracy, 3),
        "report": report,
    }

    return model, metrics


def main():
    print("Loading dataset and preparing features...")
    df, revision_encoder, status_encoder = load_and_prepare_data()

    print("Training RandomForestRegressor (Readiness Score prediction)...")
    regressor, regression_metrics = train_regression_model(df)
    print(f"  -> MAE: {regression_metrics['mae']}, R2 Score: {regression_metrics['r2_score']}")

    print("Training RandomForestClassifier (Readiness Status prediction)...")
    classifier, classification_metrics = train_classification_model(df)
    print(f"  -> Accuracy: {classification_metrics['accuracy']}")

    # Create the models folder if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save the trained models and encoders using joblib
    joblib.dump(regressor, os.path.join(MODELS_DIR, "regressor.pkl"))
    joblib.dump(classifier, os.path.join(MODELS_DIR, "classifier.pkl"))
    joblib.dump(revision_encoder, os.path.join(MODELS_DIR, "revision_encoder.pkl"))
    joblib.dump(status_encoder, os.path.join(MODELS_DIR, "status_encoder.pkl"))

    # Save the evaluation metrics so the Streamlit app can display them
    metrics_summary = {
        "regression": regression_metrics,
        "classification": classification_metrics,
        "status_labels": list(status_encoder.classes_),
    }
    joblib.dump(metrics_summary, os.path.join(MODELS_DIR, "metrics.pkl"))

    print("\nAll models and metrics saved successfully in the 'models/' folder.")


if __name__ == "__main__":
    main()
