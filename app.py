import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Exam Readiness Checker",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join("data", "student_data.csv")
MODELS_DIR = "models"

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

REVISION_OPTIONS = ["Not Started", "In Progress", "Completed"]


# ---------------------------------------------------------------------------
# CUSTOM STYLING (white background, navy headings, light blue cards)
# ---------------------------------------------------------------------------
def inject_custom_css():
    """Applies a clean, professional look to the Streamlit app."""
    st.markdown(
        """
        <style>
        .main {
            background-color: #FFFFFF;
        }
        html, body, [class*="css"] {
            font-family: "Segoe UI", "Aptos", sans-serif;
            color: #243B53;
        }
        h1, h2, h3 {
            color: #0B3D66;
            font-family: "Segoe UI", "Aptos", sans-serif;
            font-weight: 650;
        }
        h1 {
            letter-spacing: -0.03em;
        }
        [data-testid="stCaptionContainer"] {
            color: #526D82;
        }
        .metric-card {
            background: linear-gradient(135deg, #F0F7FC, #E4F0FA);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            border: 1px solid #D0E4F5;
            box-shadow: 0 3px 10px rgba(11, 61, 102, 0.06);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .metric-card:hover, .home-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 18px rgba(11, 61, 102, 0.12);
        }
        .metric-card h2 {
            color: #0B3D66;
            margin-bottom: 4px;
            font-size: 1.8rem;
        }
        .metric-card p {
            color: #4A6B85;
            margin: 0;
            font-size: 14px;
        }
        .feedback-box {
            background-color: #EAF3FB;
            border-left: 6px solid #0B3D66;
            padding: 16px;
            border-radius: 8px;
            font-size: 16px;
            color: #0B3D66;
        }
        .home-card {
            background: linear-gradient(145deg, #F4F9FD, #E7F2FA);
            border: 1px solid #D0E4F5;
            border-radius: 12px;
            padding: 20px;
            min-height: 145px;
            box-shadow: 0 3px 10px rgba(11, 61, 102, 0.06);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .home-card h3 {
            margin-top: 0;
            margin-bottom: 8px;
        }
        .home-card p {
            color: #4A6B85;
            margin-bottom: 0;
        }
        section[data-testid="stSidebar"] {
            background-color: #F5F9FC;
            color: #000000;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #000000;
        }
        [data-testid="stDeployButton"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(value, label):
    """Renders a simple light-blue metric card using HTML/CSS."""
    st.markdown(
        f"""
        <div class="metric-card">
            <h2>{value}</h2>
            <p>{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_dashboard_chart(fig):
    """Apply consistent colors and readable interactive chart styling."""
    fig.update_layout(
        template="plotly_white",
        font={"family": "Segoe UI, Aptos, sans-serif", "size": 13, "color": "#243B53"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        hoverlabel={
            "bgcolor": "#102A43",
            "font": {"family": "Segoe UI, Aptos, sans-serif", "color": "#FFFFFF"},
        },
        margin={"l": 16, "r": 16, "t": 24, "b": 16},
        legend_title_text="Readiness Status",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E6EEF5", zerolinecolor="#CBD5E1")
    fig.update_yaxes(showgrid=True, gridcolor="#E6EEF5", zerolinecolor="#CBD5E1")
    return fig


# ---------------------------------------------------------------------------
# DATA & MODEL LOADING (cached so they load only once)
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Loads the student dataset from CSV. Returns None if not found."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    """
    Loads all trained models and encoders from the models/ folder.
    Returns a dictionary of objects, or None if files are missing.
    """
    required_files = [
        "regressor.pkl", "classifier.pkl",
        "revision_encoder.pkl", "status_encoder.pkl", "metrics.pkl",
    ]
    for f in required_files:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            return None

    return {
        "regressor": joblib.load(os.path.join(MODELS_DIR, "regressor.pkl")),
        "classifier": joblib.load(os.path.join(MODELS_DIR, "classifier.pkl")),
        "revision_encoder": joblib.load(os.path.join(MODELS_DIR, "revision_encoder.pkl")),
        "status_encoder": joblib.load(os.path.join(MODELS_DIR, "status_encoder.pkl")),
        "metrics": joblib.load(os.path.join(MODELS_DIR, "metrics.pkl")),
    }


# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
def show_home_page():
    st.title("📘 AI-Based Exam Readiness Checker")
    st.subheader("Prepare with confidence. Know where you stand.")
    st.write(
        "Turn your study habits and practice results into a clear picture "
        "of your exam readiness."
    )

    st.markdown("### What would you like to explore?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="home-card"><h3>🔮 Check readiness</h3>'
            "<p>Get a personalized score and practical preparation feedback.</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="home-card"><h3>📊 Explore insights</h3>'
            "<p>Discover patterns and summaries in the student dataset.</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="home-card"><h3>🧪 View model results</h3>'
            "<p>See how the readiness prediction models performed.</p></div>",
            unsafe_allow_html=True,
        )

    st.info("Choose a page from the navigation menu to get started.")


# ---------------------------------------------------------------------------
# EXAM READINESS PREDICTION PAGE
# ---------------------------------------------------------------------------
def show_prediction_page(df, models):
    st.title("🔮 Exam Readiness Prediction")
    st.write("Enter your preparation details below to check your exam readiness.")

    if df is None or models is None:
        st.error(
            "Dataset or trained models not found. Please run "
            "`python generate_dataset.py` and `python train_models.py` "
            "before using this page."
        )
        return

    # --- Input form ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.number_input(
                "Study Hours per Day", min_value=0.0, max_value=24.0, value=4.0, step=0.5
            )
            total_chapters = st.number_input(
                "Total Chapters", min_value=1, max_value=200, value=25, step=1
            )
            chapters_completed = st.number_input(
                "Chapters Completed", min_value=0, max_value=200, value=15, step=1
            )
            mock_test_score = st.number_input(
                "Mock Test Score (0-100)", min_value=0.0, max_value=100.0, value=60.0, step=1.0
            )

        with col2:
            attendance_percentage = st.number_input(
                "Attendance Percentage (0-100)", min_value=0.0, max_value=100.0, value=75.0, step=1.0
            )
            days_left = st.number_input(
                "Days Left for Exam", min_value=1, max_value=365, value=30, step=1
            )
            confidence_level = st.slider("Confidence Level (1-10)", 1, 10, 5)
            revision_status = st.selectbox("Revision Status", REVISION_OPTIONS)

        submitted = st.form_submit_button("Predict Readiness")

    if submitted:
        # --- Validate inputs to avoid crashes ---
        if chapters_completed > total_chapters:
            st.warning(
                "Chapters Completed cannot be greater than Total Chapters. "
                "Please correct your input and try again."
            )
            return

        try:
            # Encode Revision_Status using the SAME encoder used in training
            revision_encoded = models["revision_encoder"].transform([revision_status])[0]

            # Build the input row in the exact same column order as training
            input_data = pd.DataFrame([{
                "Study_Hours_Per_Day": study_hours,
                "Total_Chapters": total_chapters,
                "Chapters_Completed": chapters_completed,
                "Mock_Test_Score": mock_test_score,
                "Attendance_Percentage": attendance_percentage,
                "Days_Left_For_Exam": days_left,
                "Confidence_Level": confidence_level,
                "Revision_Status_Encoded": revision_encoded,
            }])[FEATURE_COLUMNS]

            # --- Predictions ---
            predicted_score = float(models["regressor"].predict(input_data)[0])
            predicted_score = round(np.clip(predicted_score, 0, 100), 1)

            predicted_status_encoded = models["classifier"].predict(input_data)[0]
            predicted_status = models["status_encoder"].inverse_transform(
                [predicted_status_encoded]
            )[0]

            # --- Percentile calculation vs the dataset ---
            percentile = round(
                (df["Readiness_Score"] < predicted_score).mean() * 100, 1
            )

            # --- Display results ---
            st.markdown("### Your Results")
            r1, r2, r3 = st.columns(3)
            with r1:
                metric_card(f"{predicted_score}", "Readiness Score (0-100)")
            with r2:
                metric_card(predicted_status, "Readiness Status")
            with r3:
                metric_card(f"{percentile}%", "Percentile (vs other students)")

            # --- Personalized feedback ---
            feedback_map = {
                "Ready": "You are well prepared for the examination. Continue "
                         "revising and practicing mock tests.",
                "Almost Ready": "You are on the right track. Focus on revision "
                                "and improve your weaker areas.",
                "Needs More Preparation": "You need more preparation. Increase "
                                          "your study time and focus on "
                                          "completing important chapters.",
            }
            feedback = feedback_map.get(
                predicted_status, "Keep working on your preparation consistently."
            )

            st.markdown("### Personalized Feedback")
            st.markdown(f'<div class="feedback-box">{feedback}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong while generating your prediction: {e}")


# ---------------------------------------------------------------------------
# ANALYTICS DASHBOARD PAGE
# ---------------------------------------------------------------------------
def show_dashboard_page(df):
    st.title("📊 Analytics Dashboard")

    if df is None:
        st.error(
            "Dataset not found. Please run `python generate_dataset.py` first."
        )
        return

    st.write(
        "Explore student preparation patterns through interactive charts. "
        "Hover over points for details, or use the chart controls to zoom and explore."
    )

    # --- Summary metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(len(df), "Total Number of Students")
    with c2:
        metric_card(round(df["Study_Hours_Per_Day"].mean(), 2), "Avg Study Hours/Day")
    with c3:
        metric_card(round(df["Mock_Test_Score"].mean(), 2), "Avg Mock Test Score")
    with c4:
        metric_card(round(df["Readiness_Score"].mean(), 2), "Avg Readiness Score")

    st.markdown("---")

    # --- Readiness Status Distribution ---
    st.markdown("### Readiness Status Distribution")
    status_counts = df["Readiness_Status"].value_counts().reset_index()
    status_counts.columns = ["Readiness_Status", "Count"]
    fig1 = px.pie(
        status_counts, names="Readiness_Status", values="Count",
        color="Readiness_Status",
        color_discrete_map={
            "Ready": "#2E7D32",
            "Almost Ready": "#F9A825",
            "Needs More Preparation": "#C62828",
        },
        hole=0.4,
    )
    st.plotly_chart(
        style_dashboard_chart(fig1),
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
    )

    # --- Study Hours vs Mock Test Score ---
    st.markdown("### Study Hours vs Mock Test Score")
    fig2 = px.scatter(
        df, x="Study_Hours_Per_Day", y="Mock_Test_Score",
        color="Readiness_Status",
        color_discrete_map={
            "Ready": "#2E7D32",
            "Almost Ready": "#F9A825",
            "Needs More Preparation": "#C62828",
        },
        labels={"Study_Hours_Per_Day": "Study Hours per Day", "Mock_Test_Score": "Mock Test Score"},
        hover_data=["Readiness_Score", "Attendance_Percentage"],
    )
    fig2.update_traces(marker={"size": 10, "opacity": 0.78, "line": {"width": 0.5, "color": "#FFFFFF"}})
    st.plotly_chart(
        style_dashboard_chart(fig2),
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
    )

    # --- Attendance vs Mock Test Score ---
    st.markdown("### Attendance vs Mock Test Score")
    fig3 = px.scatter(
        df, x="Attendance_Percentage", y="Mock_Test_Score",
        color="Readiness_Status",
        color_discrete_map={
            "Ready": "#2E7D32",
            "Almost Ready": "#F9A825",
            "Needs More Preparation": "#C62828",
        },
        labels={"Attendance_Percentage": "Attendance (%)", "Mock_Test_Score": "Mock Test Score"},
        hover_data=["Readiness_Score", "Study_Hours_Per_Day"],
    )
    fig3.update_traces(marker={"size": 10, "opacity": 0.78, "line": {"width": 0.5, "color": "#FFFFFF"}})
    st.plotly_chart(
        style_dashboard_chart(fig3),
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
    )

    # --- Chapter Completion vs Readiness Score ---
    st.markdown("### Chapter Completion vs Readiness Score")
    df_temp = df.copy()
    df_temp["Completion_Ratio"] = (
        df_temp["Chapters_Completed"] / df_temp["Total_Chapters"] * 100
    )
    fig4 = px.scatter(
        df_temp, x="Completion_Ratio", y="Readiness_Score",
        color="Readiness_Status",
        color_discrete_map={
            "Ready": "#2E7D32",
            "Almost Ready": "#F9A825",
            "Needs More Preparation": "#C62828",
        },
        labels={"Completion_Ratio": "Chapter Completion (%)", "Readiness_Score": "Readiness Score"},
        hover_data=["Chapters_Completed", "Total_Chapters", "Mock_Test_Score"],
    )
    fig4.update_traces(marker={"size": 10, "opacity": 0.78, "line": {"width": 0.5, "color": "#FFFFFF"}})
    st.plotly_chart(
        style_dashboard_chart(fig4),
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
    )


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE PAGE
# ---------------------------------------------------------------------------
def show_model_performance_page(models):
    st.title("🧪 Model Performance")

    if models is None:
        st.error(
            "Trained models not found. Please run `python train_models.py` first."
        )
        return

    metrics = models["metrics"]

    st.markdown("### Regression Model — RandomForestRegressor")
    st.write("Predicts the numeric Readiness Score (0-100).")
    r1, r2 = st.columns(2)
    with r1:
        metric_card(metrics["regression"]["mae"], "Mean Absolute Error (MAE)")
    with r2:
        metric_card(metrics["regression"]["r2_score"], "R² Score")

    st.markdown("---")

    st.markdown("### Classification Model — RandomForestClassifier")
    st.write("Predicts the Readiness Status category.")
    metric_card(metrics["classification"]["accuracy"], "Accuracy")

    st.markdown("#### Classification Report")
    report_df = pd.DataFrame(metrics["classification"]["report"]).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    inject_custom_css()

    df = load_dataset()
    models = load_models()

    st.sidebar.title("📘 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Exam Readiness Prediction",
            "Analytics Dashboard",
            "Model Performance",
        ],
    )

    if page == "Home":
        show_home_page()
    elif page == "Exam Readiness Prediction":
        show_prediction_page(df, models)
    elif page == "Analytics Dashboard":
        show_dashboard_page(df)
    elif page == "Model Performance":
        show_model_performance_page(models)
if __name__ == "__main__":
    main()
