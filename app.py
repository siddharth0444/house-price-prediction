import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('model/house_price_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Header ────────────────────────────────────────────────────
st.title("🏠 California House Price Predictor")
st.markdown("**Built by Siddharth Rajak** | ML Project #1")
st.markdown("Enter the details below to predict the median house value.")
st.divider()

# ── Input form ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    MedInc     = st.slider("💰 Median Income (×$10,000)", 0.5, 15.0, 3.5, 0.1,
                            help="Median income of block group in tens of thousands of dollars")
    HouseAge   = st.slider("🏗️ House Age (years)", 1, 52, 20,
                            help="Median age of houses in the block")
    AveRooms   = st.slider("🛏️ Average Rooms", 1.0, 15.0, 5.0, 0.1,
                            help="Average number of rooms per household")
    AveBedrms  = st.slider("🛏️ Average Bedrooms", 0.5, 5.0, 1.0, 0.1,
                            help="Average number of bedrooms per household")

with col2:
    Population = st.number_input("👥 Block Population", 10, 30000, 1500,
                                  help="Population of the block group")
    AveOccup   = st.slider("👨‍👩‍👧 Average Occupants", 1.0, 10.0, 3.0, 0.1,
                            help="Average number of household members")
    Latitude   = st.slider("🌐 Latitude", 32.5, 42.0, 37.0, 0.1,
                            help="Block latitude (Northern CA ≈ 38-42, Southern CA ≈ 32-36)")
    Longitude  = st.slider("🌐 Longitude", -124.5, -114.0, -119.0, 0.1,
                            help="Block longitude (Coastal ≈ -122 to -124)")

st.divider()

# ── Prediction ────────────────────────────────────────────────
if st.button("🔮 Predict House Price", use_container_width=True, type="primary"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                           Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(features)[0]
    price_usd  = prediction * 100_000

    st.success(f"### 🏡 Estimated Median House Value")
    st.metric(label="Predicted Price", value=f"${price_usd:,.0f}")

    # Confidence range ±10%
    low  = price_usd * 0.90
    high = price_usd * 1.10
    st.caption(f"Estimated range: **${low:,.0f}** – **${high:,.0f}**")

    st.divider()

    # Input summary
    with st.expander("📋 View Your Input Summary"):
        import pandas as pd
        housing = fetch_california_housing()
        summary = pd.DataFrame({
            'Feature': housing.feature_names,
            'Your Input': [MedInc, HouseAge, AveRooms, AveBedrms,
                           Population, AveOccup, Latitude, Longitude]
        })
        st.dataframe(summary, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:13px;'>
    Built with ❤️ by <b>Siddharth Rajak</b> | 
    Powered by Random Forest & Scikit-learn |
    <a href='https://github.com/siddharth0444' target='_blank'>GitHub</a> · 
    <a href='https://linkedin.com/in/siddharthrajak' target='_blank'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
