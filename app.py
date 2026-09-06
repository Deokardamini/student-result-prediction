import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# Try to import streamlit-confetti; fallback cleanly if not installed
try:
    from streamlit_confetti import confetti
    HAS_CONFETTI = True
except ImportError:
    HAS_CONFETTI = False

# Page Configuration
st.set_page_config(
    page_title="Academic Predictor AI",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    /* Gradient Background for App Header */
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .header-box h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Metric Card Styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Predict Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 30px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 12px rgba(56, 239, 125, 0.3);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(56, 239, 125, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Cache model loading for fast performance
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Application Header
st.markdown("""
    <div class="header-box">
        <h1>🎓 Academic Performance Predictor</h1>
        <p>Input your 6 core subject marks to predict evaluation outcomes using your KNN Model</p>
    </div>
""", unsafe_allow_html=True)

# Main Form Layout
st.subheader("📝 Enter Subject Marks (Out of 100)")

col1, col2 = st.columns(2)

with col1:
    hindi = st.number_input("Hindi", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    english = st.number_input("English", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    science = st.number_input("Science", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

with col2:
    maths = st.number_input("Maths", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    history = st.number_input("History", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
    geography = st.number_input("Geography", min_value=0.0, max_value=100.0, value=72.0, step=1.0)

# Automatic total score calculation
total_marks = hindi + english + science + maths + history + geography

st.markdown("---")

# Prediction trigger section
st.write("### Ready to Predict?")
predict_btn = st.button("🚀 Generate Prediction")

if predict_btn:
    # 1. Processing / Loading Effect
    with st.spinner("Analyzing marks and running KNN inference..."):
        time.sleep(1.2)  # Simulated brief loading animation
        
        # Prepare input matching feature names: ['Hindi', 'English', 'Science', 'Maths', 'History', 'Geograpgy', 'Total']
        input_data = pd.DataFrame([{
            'Hindi': hindi,
            'English': english,
            'Science': science,
            'Maths': maths,
            'History': history,
            'Geograpgy': geography,
            'Total': total_marks
        }])

        prediction = model.predict(input_data)[0]

    # 2. Trigger Visual Celebratory Effects
    st.balloons()
    if HAS_CONFETTI:
        confetti()

    # 3. Display Results
    st.success("Analysis Complete!")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"""
            <div class="metric-card">
                <h4>Calculated Total</h4>
                <h2 style="color: #667eea;">{total_marks:.1f} / 600</h2>
            </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
            <div class="metric-card">
                <h4>Model Prediction Result</h4>
                <h2 style="color: #38ef7d;">{prediction}</h2>
            </div>
        """, unsafe_allow_html=True)
