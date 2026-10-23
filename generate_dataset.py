"""
generate_dataset.py
--------------------
This script creates a SYNTHETIC dataset of 200 student records for the
"AI-Based Exam Readiness Checker" project.

It generates realistic-looking values for each input feature (study hours,
chapters completed, mock test score, attendance, etc.), then calculates a
Readiness Score (0-100) using a weighted formula plus some random noise so
that the data still has learnable, realistic patterns for Machine Learning.

Finally, it saves everything into: data/student_data.csv

Run this file first, before training the models:
    python generate_dataset.py
"""

import pandas as pd
import numpy as np
import os

# Set a random seed so the dataset is the same every time we run this script
np.random.seed(42)

# Number of synthetic student records to create
NUM_STUDENTS = 200


def generate_student_data(num_students):
    """
    Generates synthetic student data for all input features.
    Returns a Pandas DataFrame.
    """

    # 1. Study Hours per Day (realistic range: 0.5 to 10 hours)
    study_hours = np.round(np.random.uniform(0.5, 10, num_students), 1)

    # 2. Total Chapters (fixed range depending on subject, e.g., 10 to 40)
    total_chapters = np.random.randint(10, 41, num_students)

    # 3. Chapters Completed (must be <= total_chapters)
    chapters_completed = np.array([
        np.random.randint(0, tc + 1) for tc in total_chapters
    ])

    # 4. Mock Test Score (0 to 100)
    mock_test_score = np.round(np.random.uniform(20, 100, num_students), 1)

    # 5. Attendance Percentage (40% to 100%)
    attendance_percentage = np.round(np.random.uniform(40, 100, num_students), 1)

    # 6. Days Left for Exam (1 to 90 days)
    days_left = np.random.randint(1, 91, num_students)

    # 7. Confidence Level (1 to 10 scale)
    confidence_level = np.random.randint(1, 11, num_students)

    # 8. Revision Status (categorical: Not Started, In Progress, Completed)
    revision_status = np.random.choice(
        ["Not Started", "In Progress", "Completed"],
        num_students,
        p=[0.25, 0.45, 0.30]
    )

    # Build the DataFrame with all the raw input features
    df = pd.DataFrame({
        "Study_Hours_Per_Day": study_hours,
        "Total_Chapters": total_chapters,
        "Chapters_Completed": chapters_completed,
        "Mock_Test_Score": mock_test_score,
        "Attendance_Percentage": attendance_percentage,
        "Days_Left_For_Exam": days_left,
        "Confidence_Level": confidence_level,
        "Revision_Status": revision_status,
    })

    return df


def calculate_readiness_score(df):
    """
    Calculates a Readiness Score (0-100) for each student using a
    weighted combination of the input features. This mimics how a
    real exam-readiness score might be estimated, and gives the
    Machine Learning models a realistic pattern to learn from.
    """

    # Chapter completion ratio (0 to 1)
    completion_ratio = df["Chapters_Completed"] / df["Total_Chapters"]

    # Convert Revision_Status into a numeric weight
    revision_weight_map = {"Not Started": 0.0, "In Progress": 0.5, "Completed": 1.0}
    revision_weight = df["Revision_Status"].map(revision_weight_map)

    # Days left is inversely useful -- fewer days left slightly reduces score
    # unless preparation is already strong. We normalize it (more days = more time to prepare)
    days_left_factor = np.clip(df["Days_Left_For_Exam"] / 90, 0, 1)

    # Weighted formula (weights sum to 1.0 across major factors)
    raw_score = (
        (df["Study_Hours_Per_Day"] / 10) * 20 +       # up to 20 points
        completion_ratio * 20 +                        # up to 20 points
        (df["Mock_Test_Score"] / 100) * 25 +            # up to 25 points
        (df["Attendance_Percentage"] / 100) * 10 +      # up to 10 points
        revision_weight * 10 +                          # up to 10 points
        (df["Confidence_Level"] / 10) * 10 +            # up to 10 points
        days_left_factor * 5                             # up to 5 points
    )

    # Add small random noise to simulate real-world unpredictability
    noise = np.random.normal(0, 3, len(df))
    final_score = raw_score + noise

    # Keep the score within a valid 0-100 range
    final_score = np.clip(final_score, 0, 100)

    return np.round(final_score, 1)


def calculate_readiness_status(score):
    """
    Converts a numeric Readiness Score into a Readiness Status category.
    """
    if score >= 75:
        return "Ready"
    elif score >= 50:
        return "Almost Ready"
    else:
        return "Needs More Preparation"


def main():
    """
    Main function that generates the dataset, calculates the target
    columns, and saves the final CSV file.
    """
    print("Generating synthetic student dataset...")

    # Step 1: Generate raw input features
    df = generate_student_data(NUM_STUDENTS)

    # Step 2: Calculate the Readiness Score (regression target)
    df["Readiness_Score"] = calculate_readiness_score(df)

    # Step 3: Calculate the Readiness Status (classification target)
    df["Readiness_Status"] = df["Readiness_Score"].apply(calculate_readiness_status)

    # Step 4: Create the "data" folder if it does not already exist
    os.makedirs("data", exist_ok=True)

    # Step 5: Save the dataset as a CSV file
    output_path = os.path.join("data", "student_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset generated successfully with {len(df)} student records.")
    print(f"Saved to: {output_path}")
    print("\nReadiness Status distribution:")
    print(df["Readiness_Status"].value_counts())


if __name__ == "__main__":
    main()
