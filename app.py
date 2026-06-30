import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Travel Destination Predictor", page_icon="🌴")

# 1. Load the new assets
@st.cache_resource
def load_assets():
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_encoder.pkl', 'rb') as f:
        fe = pickle.load(f)
    return model, fe

model, feature_encoder = load_assets()

st.title("🌴 Travel Destination Predictor")
st.write("Enter your preferences to predict your ideal vacation spot.")
st.markdown("---")

# 2. User Inputs
budget = st.number_input("💵 Total Budget (INR)", min_value=5000.0, max_value=300000.0, value=25000.0)
duration = st.slider("📅 Trip Duration (Days)", min_value=1, max_value=30, value=5)
rating = st.slider("⭐ Minimum Destination Rating", min_value=1.0, max_value=5.0, value=4.5)

travel_type = st.selectbox("👥 Travel Type", list(feature_encoder.categories_[0]))
season = st.selectbox("🌤️ Season", list(feature_encoder.categories_[1]))
transport = st.selectbox("✈️ Transport Mode", list(feature_encoder.categories_[2]))
hotel_type = st.selectbox("🏨 Hotel Accommodation", list(feature_encoder.categories_[3]))

if st.button("🚀 Find My Perfect Destination", use_container_width=True):
    # Organize selections into a DataFrame
    input_data = pd.DataFrame([{
        'Budget': budget,
        'Duration_Days': duration,
        'Rating': rating,
        'Travel_Type': travel_type,
        'Season': season,
        'Transport': transport,
        'Hotel_Type': hotel_type
    }])
    
    # Process inputs using the feature encoder
    categorical_cols = ['Travel_Type', 'Season', 'Transport', 'Hotel_Type']
    input_data[categorical_cols] = feature_encoder.transform(input_data[categorical_cols])
    
    # Predict directly (This returns the actual string name now!)
    prediction = model.predict(input_data)[0]
    
    # Calculate Confidence
    probabilities = model.predict_proba(input_data)[0]
    confidence = np.max(probabilities) * 100
    
    # Display Output
    st.markdown("### 🗺️ Recommended Destination:")
    st.success(f"🎉 Your ideal destination is: **{prediction}**")
    st.info(f"💡 Prediction Confidence: **{confidence:.2f}%**")