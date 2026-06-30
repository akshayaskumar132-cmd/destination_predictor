import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration & Styling
st.set_page_config(page_title="Travel Destination Predictor", page_icon="🌴", layout="centered")

# 2. Load the trained Logistic Regression model
@st.cache_resource
def load_model():
    with open('logistic_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# 3. Application Title
st.title("🌴 Logistic Regression Travel Predictor")
st.write("Using a trained Logistic Regression classifier to map your budget and lifestyle preferences to the ultimate getaway destination.")
st.markdown("---")

# 4. Input Fields
st.header("✨ Customize Your Ideal Trip")

# Numerical Inputs
budget = st.number_input("💵 Total Budget (INR)", min_value=5000.0, max_value=300000.0, value=25000.0, step=1000.0)
duration = st.slider("📅 Trip Duration (Days)", min_value=1, max_value=30, value=5)
rating = st.slider("⭐ Minimum Destination Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

# Categorical Dropdowns
travel_type = st.selectbox("👥 Companionship / Travel Type", ["Solo", "Couple", "Family", "Friends"])
season = st.selectbox("🌤️ Preferred Season", ["Summer", "Monsoon", "Winter"])
transport = st.selectbox("✈️ Transport Mode", ["Bus", "Car", "Train", "Flight"])
hotel_type = st.selectbox("🏨 Hotel Accommodation Style", ["Budget", "Standard", "Luxury"])

# 5. Mappings for Preprocessing (Matches Google Colab exactly)
travel_map = {'Solo': 0, 'Couple': 1, 'Family': 2, 'Friends': 3}
season_map = {'Summer': 0, 'Monsoon': 1, 'Winter': 2}
transport_map = {'Bus': 0, 'Car': 1, 'Train': 2, 'Flight': 3}
hotel_map = {'Budget': 0, 'Standard': 1, 'Luxury': 2}

# 6. Make Predictions
if st.button("🚀 Find My Perfect Destination", use_container_width=True):
    # Convert categorical inputs to numerical representations
    features = np.array([[
        budget,
        duration,
        rating,
        travel_map[travel_type],
        season_map[season],
        transport_map[transport],
        hotel_map[hotel_type]
    ]])
    
    # Run prediction and get probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = np.max(probabilities) * 100
    
    # Display Results
    st.markdown("### 🗺️ Recommended Destination:")
    st.success(f"🎉 **{prediction}**")
    st.info(f"💡 Model Confidence: **{confidence:.2f}%**")