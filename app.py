import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(
    page_title="Academic Performance Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for Vertical Layout, Shadows, and Aesthetics
st.markdown(
    """
    <style>
    /* Main Background Accent */
    .stApp {
        background-color: #f4f6f9;
    }

    /* Card Shadow Box Wrapper */
    .custom-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }

    /* Header Design */
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        padding: 2rem;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.2);
        margin-bottom: 2rem;
    }

    .header-card h1 {
        color: #ffffff !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Result Cards */
    .res-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-top: 4px solid #2a5298;
    }

    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 12px rgba(56, 239, 125, 0.3);
        transition: all 0.3s ease;
    }

    div.stButton > button:first-child:hover {
        box-shadow: 0 6px 18px rgba(56, 239, 125, 0.5);
        transform: translateY(-2px);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load Trained KNN Model
@st.cache_resource
def load_model():
  with open("model.pkl", "rb") as f:
    return pickle.load(f)


model = load_model()

# Header Section
st.markdown(
    """
    <div class="header-card">
        <h1>🎓 Academic Performance Predictor</h1>
        <p>Enter individual subject marks vertically below to evaluate student outcomes.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Vertical Input Form inside Shadow Card
st.markdown(
    '<div class="custom-card"><h3>📝 Enter Subject Marks (0 - 100)</h3>',
    unsafe_allow_html=True,
)

hindi = st.number_input(
    "Hindi Marks", min_value=0.0, max_value=100.0, value=75.0, step=1.0
)
english = st.number_input(
    "English Marks", min_value=0.0, max_value=100.0, value=80.0, step=1.0
)
science = st.number_input(
    "Science Marks", min_value=0.0, max_value=100.0, value=70.0, step=1.0
)
maths = st.number_input(
    "Maths Marks", min_value=0.0, max_value=100.0, value=85.0, step=1.0
)
history = st.number_input(
    "History Marks", min_value=0.0, max_value=100.0, value=65.0, step=1.0
)
geography = st.number_input(
    "Geography Marks", min_value=0.0, max_value=100.0, value=72.0, step=1.0
)

st.markdown("</div>", unsafe_allow_html=True)

# Calculate Total Score Automatically
total_marks = hindi + english + science + maths + history + geography

# Action Button
predict_btn = st.button("🚀 Calculate & Predict Result")

if predict_btn:
  with st.spinner("Processing evaluation through KNN model..."):
    time.sleep(1)

    # DataFrame structure matching feature names expected by your model
    input_data = pd.DataFrame([{
        "Hindi": hindi,
        "English": english,
        "Science": science,
        "Maths": maths,
        "History": history,
        "Geograpgy": geography,
        "Total": total_marks,
    }])

    prediction = model.predict(input_data)[0]

  # Visual Effects
  st.balloons()

  # Safe Canvas Confetti Trigger via JavaScript
  components.html(
      """
      <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
      <script>
          confetti({
              particleCount: 120,
              spread: 80,
              origin: { y: 0.6 }
          });
      </script>
  """,
      height=0,
  )

  # Display Results inside Styled Cards
  st.markdown("---")
  st.markdown("### 📊 Prediction Results")

  col_res1, col_res2 = st.columns(2)

  with col_res1:
    st.markdown(
        f"""
          <div class="res-box">
              <p style="color: #64748b; margin-bottom: 5px; font-weight: 600;">Calculated Total</p>
              <h2 style="color: #1e3c72; margin: 0;">{total_marks:.1f} / 600</h2>
          </div>
      """,
        unsafe_allow_html=True,
    )

  with col_res2:
    st.markdown(
        f"""
          <div class="res-box">
              <p style="color: #64748b; margin-bottom: 5px; font-weight: 600;">Predicted Outcome</p>
              <h2 style="color: #11998e; margin: 0;">{prediction}</h2>
          </div>
      """,
        unsafe_allow_html=True,
    )
