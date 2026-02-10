import streamlit as st
import numpy as np
import joblib
import base64
import pandas as pd

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="NBA Career Longevity Predictor",
    layout="centered"
)

# ==============================
# Background Image Function
# ==============================
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.65);
            padding: 2rem;
            border-radius: 15px;
        }}

        h1, h2, h3, p, label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Background image
set_background("background.png")

# ==============================
# Load Trained Model
# ==============================
model = joblib.load("career_longevity_gbt_model.pkl")

# ==============================
# App Title & Description
# ==============================
st.title("🏀 NBA Career Longevity Predictor")
st.write(
    "Predict both whether an NBA player will have a career lasting **≥ 5 years** "
    "and estimate the likely number of years played based on early-career performance."
)

st.divider()

# ==============================
# Slider Inputs
# ==============================
games_played = st.slider("Games Played (GP)", 0, 82, 50)
minutes = st.slider("Average Minutes Played", 0.0, 48.0, 20.0)
points = st.slider("Points Per Game", 0.0, 40.0, 10.0)
rebounds = st.slider("Rebounds Per Game", 0.0, 20.0, 5.0)
assists = st.slider("Assists Per Game", 0.0, 15.0, 3.0)
steals = st.slider("Steals Per Game", 0.0, 5.0, 1.0)
blocks = st.slider("Blocks Per Game", 0.0, 5.0, 0.5)
turnovers = st.slider("Turnovers Per Game", 0.0, 10.0, 2.0)

st.divider()

# ==============================
# Prediction
# ==============================
if st.button("🔮 Predict Career Longevity"):

    # Arrange features in SAME order as training
    input_data = np.array([[games_played, minutes, points, rebounds,
                            assists, steals, blocks, turnovers]])

    # Predict target (≥5 years)
    target_prediction = model.predict(input_data)[0]
    prob_long = model.predict_proba(input_data)[0][1]

    # Estimate career length (heuristic based on probability)
    if target_prediction == 1:
        estimated_years = int(5 + prob_long * 10)  # 5–15 years
        st.success("✅ **Prediction: Career ≥ 5 Years**")
    else:
        estimated_years = max(1, int(prob_long * 5))  # 1–4 years
        st.error("❌ **Prediction: Career < 5 Years**")

    st.subheader("📊 Prediction Details")
    st.write(f"**Estimated Career Length:** {estimated_years} years")
    st.write(f"**Model Confidence:** {prob_long:.2%}")

    st.caption(
        "⚠️ Career length is an estimate derived from prediction probability, "
        "not a direct regression output."
    )

# ==============================
# Footer
# ==============================
st.divider()
st.caption(
    "Model: Gradient Boosting Classifier | "
    "Dataset Source: Kaggle NBA Players Dataset | "
    "For academic use only"
)
