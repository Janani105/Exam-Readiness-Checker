# AI-Based Exam Readiness Checker Using Data Analytics and Machine Learning

A beginner-friendly Streamlit web application that evaluates a student's exam
preparation using Data Analytics and Machine Learning, built for a 15-day
internship in the domain of **AI with Data Analytics**.

---

## 📌 Project Overview

This project predicts a student's **Exam Readiness Score (0–100)** and
**Readiness Status** (Ready / Almost Ready / Needs More Preparation) based on
their study habits, mock test performance, attendance, and revision progress.
It combines Data Analytics (Pandas, NumPy, Plotly) with Machine Learning
(Scikit-learn) inside a simple Streamlit web app.

## 🎯 Objective

To build a Data Analytics and Machine Learning powered web application that
evaluates a student's exam preparation level and provides personalized,
actionable feedback to help them improve before the exam.

## 🧰 Technologies Used

- **Python** — core language
- **Pandas** & **NumPy** — data generation, cleaning, and analytics
- **Scikit-learn** — RandomForestRegressor & RandomForestClassifier
- **Plotly** — interactive charts
- **Streamlit** — web application interface

No other frameworks, databases, or external APIs are used.

## 📂 Project Structure

```
exam_readiness_checker/
│
├── app.py                  # Main Streamlit application (4 pages)
├── generate_dataset.py     # Creates the synthetic dataset (200 students)
├── train_models.py         # Trains and saves the ML models
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation (this file)
│
├── data/
│   └── student_data.csv    # Generated dataset (created after running generate_dataset.py)
│
└── models/                 # Trained models & metrics (created after running train_models.py)
    ├── regressor.pkl
    ├── classifier.pkl
    ├── revision_encoder.pkl
    ├── status_encoder.pkl
    └── metrics.pkl
```

## 🖥️ Application Pages

1. **Home** — project title
2. **Exam Readiness Prediction** — enter your details and get a Readiness
   Score, Readiness Status, percentile, and personalized feedback
3. **Analytics Dashboard** — dataset statistics and interactive Plotly charts
4. **Model Performance** — regression and classification evaluation metrics

## ⚙️ Setup & Run Instructions (VS Code / Local Machine)

### 1. Install the required packages
```bash
pip install -r requirements.txt
```

### 2. Generate the synthetic dataset
```bash
python generate_dataset.py
```
This creates `data/student_data.csv` with 200 synthetic student records.

### 3. Train the Machine Learning models
```bash
python train_models.py
```
This trains the RandomForestRegressor and RandomForestClassifier, evaluates
them, and saves the trained models into the `models/` folder.

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
This opens the app in your browser (usually at `http://localhost:8501`).

## 📊 Machine Learning Models

| Model | Type | Predicts |
|---|---|---|
| RandomForestRegressor | Regression | Readiness Score (0–100) |
| RandomForestClassifier | Classification | Readiness Status (Ready / Almost Ready / Needs More Preparation) |

**Regression metrics:** Mean Absolute Error (MAE), R² Score
**Classification metrics:** Accuracy, Classification Report

## 🔮 Future Scope

- Subject-wise readiness tracking
- Tracking readiness improvement over time
- Personalized study plan recommendations
- Expansion with real (consented) student data

## 📝 Notes

- The dataset is **synthetic** and generated purely for demonstration and
  learning purposes.
- Always run `generate_dataset.py` and `train_models.py` **before**
  `streamlit run app.py`, otherwise the app will show a friendly warning
  message instead of crashing.
