import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page setup for a wide, beautiful dashboard layout
st.set_page_config(page_title="AI Travel Companion", page_icon="✈️", layout="wide")

# Custom CSS styling for a cleaner look
st.markdown("""
    <style>
    .main-title { font-size: 45px; font-weight: 800; color: #ff4b4b; text-align: center; margin-top: 20px; }
    .subtitle { font-size: 18px; text-align: center; color: #666666; margin-bottom: 30px; }
    .prediction-box { background-color: #f0f2f6; padding: 25px; border-radius: 12px; border-left: 6px solid #ff4b4b; margin-top: 15px; margin-bottom: 25px; }
    .welcome-card { background-color: #ffffff; padding: 40px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    </style>
""", unsafe_allow_html=True)

# 1. Load Core Model Assets
@st.cache_resource
def load_assets():
    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_encoder.pkl', 'rb') as f:
        fe = pickle.load(f)
    return model, fe

model, feature_encoder = load_assets()

# 2. Initialize Page Navigation Session States
if 'page' not in st.session_state:
    st.session_state.page = 1

def go_to_page(page_num):
    st.session_state.page = page_num
    st.rerun()

# =====================================================================
# PAGE 1: WELCOME SCREEN
# =====================================================================
if st.session_state.page == 1:
    st.markdown("<div class='main-title'>🌴 Welcome to your AI Travel Companion</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Find your perfect getaway based on advanced data analytics</div>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_mid = st.columns([1, 2, 1])[1]
    
    with col_mid:
        st.markdown("""
        <div class='welcome-card'>
            <h3> Let's help you pack your bags!</h3>
            <p>Our Logistic Regression engine will analyze your budget, travel logistics, and season preferences to accurately recommend the absolute best destination match from our travel logs.</p>
            <br>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Get Started 🚀", use_container_width=True):
            go_to_page(2)

# =====================================================================
# PAGE 2: USER INPUT FORM
# =====================================================================
elif st.session_state.page == 2:
    st.markdown("<div class='main-title'>🎒 Tell Us About Your Dream Trip</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Step 2 of 3: Configure your lifestyle and budget preferences</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("✨ Core Trip Preferences")
        st.session_state.budget = st.number_input("💵 Total Budget (INR)", min_value=5000.0, max_value=300000.0, value=25000.0, step=5000.0)
        st.session_state.duration = st.slider("📅 Trip Duration (Days)", min_value=1, max_value=30, value=5)
        st.session_state.rating = st.slider("⭐ Target Destination Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

    with col2:
        st.subheader("✈️ Logistics Configuration")
        st.session_state.travel_type = st.selectbox("👥 Companion Type", list(feature_encoder.categories_[0]))
        st.session_state.season = st.selectbox("🌤️ Preferred Season", list(feature_encoder.categories_[1]))
        st.session_state.transport = st.selectbox("✈️ Mode of Transport", list(feature_encoder.categories_[2]))
        st.session_state.hotel_type = st.selectbox("🏨 Hotel Luxury Level", list(feature_encoder.categories_[3]))
        
    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        if st.button("⬅️ Back to Welcome Screen", use_container_width=True):
            go_to_page(1)
    with col_b2:
        if st.button("Find My Destination 🚀", use_container_width=True):
            go_to_page(3)

# =====================================================================
# PAGE 3: RECOMMENDATION & DATA ANALYSIS
# =====================================================================
elif st.session_state.page == 3:
    st.markdown("<div class='main-title'>🗺️ Your Perfect Match Revealed</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Step 3 of 3: AI prediction report generated successfully</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Check if user accidentally jumped pages without assigning configurations
    if 'budget' not in st.session_state:
        st.error("Missing input metrics! Please navigate back to configure your preferences.")
        if st.button("Go to Preferences Step", use_container_width=True):
            go_to_page(2)
    else:
        # Organize data frames directly out of Session State storage
        input_data = pd.DataFrame([{
            'Budget': st.session_state.budget,
            'Duration_Days': st.session_state.duration,
            'Rating': st.session_state.rating,
            'Travel_Type': st.session_state.travel_type,
            'Season': st.session_state.season,
            'Transport': st.session_state.transport,
            'Hotel_Type': st.session_state.hotel_type
        }])
        
        # Preprocessing Categories
        categorical_cols = ['Travel_Type', 'Season', 'Transport', 'Hotel_Type']
        input_data[categorical_cols] = feature_encoder.transform(input_data[categorical_cols])
        
        # Calculate Math Probabilities
        probabilities = model.predict_proba(input_data)[0]
        
        # Map indices to locations cleanly
        raw_prediction = model.predict(input_data)[0]
        try:
            fixed_classes = ['Dubai', 'Goa', 'Jaipur', 'Kerala', 'Maldives', 'Manali', 'Munnar', 'Ooty']
            text_prediction = fixed_classes[int(raw_prediction)]
            display_classes = fixed_classes
        except:
            text_prediction = str(raw_prediction)
            display_classes = list(model.classes_)

        confidence = np.max(probabilities) * 100
        
        # Display Primary Result Card
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown(f"## 🔥 Top Match: **{text_prediction}**")
        st.markdown(f"Confidence score calculated by model: **{confidence:.2f}%**")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data Chart Visualization Breakdown
        st.markdown("### 📊 Alternative Matches Analysis")
        st.write("This chart reveals how much individual weights influenced other locations behind the scenes:")
        
        prob_df = pd.DataFrame({
            'Destination': display_classes,
            'Match Strength (%)': [round(p * 100, 2) for p in probabilities]
        }).sort_values(by='Match Strength (%)', ascending=False)
        
        st.bar_chart(prob_df.set_index('Destination'))
        
        st.markdown("<br><hr>", unsafe_allow_html=True)
        if st.button("🔄 Plan Another Trip / Change Inputs", use_container_width=True):
            go_to_page(2)