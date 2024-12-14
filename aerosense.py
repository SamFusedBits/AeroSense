import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai

# Configure API
genai.configure(api_key=st.secrets["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to load pre-trained models and scaler
@st.cache_resource
def load_models_and_scaler():
    nn_model = keras.models.load_model('best_model.keras')
    xgb_model = joblib.load('xgboost_model.joblib')
    rf_model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('minmax_scaler.joblib')
    return nn_model, xgb_model, rf_model, scaler

# AQI Prediction Function
def predict_aqi(input_values, nn_model, xgb_model, rf_model, scaler):
    input_df = pd.DataFrame([input_values])
    input_scaled = scaler.transform(input_df)
    nn_pred = nn_model.predict(input_scaled).flatten()[0]
    xgb_pred = xgb_model.predict(input_scaled)[0]
    rf_pred = rf_model.predict(input_scaled)[0]
    ensemble_pred = (nn_pred + xgb_pred + rf_pred) / 3
    return ensemble_pred

# Construct Gemini Prompt
def construct_prompt(aqi, answers, follow_up_query=None, detailed=False):
    prompt = f"""
    The current Air Quality Index (AQI) is {aqi:.2f}, indicating a level of air pollution. Below are the key environmental factors:

    - Traffic Intensity: {'Very High' if answers['traffic'] == 'a' else 'Moderate' if answers['traffic'] == 'b' else 'Low'}
    - Proximity of Industrial Zones: {'Within 5 km' if answers['industrial'] == 'a' else 'Within 10 km' if answers['industrial'] == 'b' else 'More than 10 km away'}
    - Green Cover: {'Sparse' if answers['green_cover'] == 'a' else 'Moderate' if answers['green_cover'] == 'b' else 'Dense'}
    - Climate Change Mitigation Initiatives: {'None' if answers['climate_initiatives'] == 'a' else 'A few' if answers['climate_initiatives'] == 'b' else 'Actively implemented'}
    - Zoning and Development: {'Mixed zoning' if answers['zoning'] == 'a' else 'No mixed zoning'}

    - Significant Environmental Challenge: {answers['environmental_challenge']}
    - Recent Initiatives or Unique Aspects: {answers['area_initiatives']}

    """
    if detailed and follow_up_query:
        prompt = f"{prompt}\n\nAdditional Query: {follow_up_query}\nProvide further insights or details in response."
    else:
        prompt = f"{prompt}\n\nBased on this context, provide concise and actionable recommendations to improve air quality, enhance sustainability, and promote public health. Keep the suggestions brief and easy to understand for quick implementation."
    return prompt

# Main Application
def main():
    st.set_page_config(page_title="AeroSense", page_icon="🌍", layout="wide")

    st.markdown(
    """
    <div style="text-align: center;">
        <h1>🌍 AeroSense: Air Quality Intelligence</h1>
        <p>Transforming cities with AI-driven recommendations to combat air pollution, improve sustainability, and enhance public health.</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
    )

    nn_model, xgb_model, rf_model, scaler = load_models_and_scaler()

    # Initialize session state for workflow tracking
    if "aqi_predicted" not in st.session_state:
        st.session_state.aqi_predicted = False
        st.session_state.predicted_aqi = None
        st.session_state.answers = {}
        st.session_state.option = "Predict AQI from Input Parameters"
        st.session_state.show_detailed_insights = False
    
    # User Selection: Predict or Enter AQI
    option = st.radio("How would you like to proceed?", 
                      options=["Predict AQI from Input Parameters", "Manually Enter AQI Value"], 
                      index=0)

    # Reset session state if option changes
    if option != st.session_state.option:
        st.session_state.aqi_predicted = False
        st.session_state.predicted_aqi = None
        st.session_state.answers = {}
        st.session_state.option = option
    
    if option == "Predict AQI from Input Parameters" and not st.session_state.aqi_predicted:
        with st.expander("Input Parameters for AQI Prediction", expanded=not st.session_state.aqi_predicted):
            # Input for AQI prediction
            col1, col2, col3 = st.columns(3)
            with col1:
                co_gt = st.number_input("CO(GT) - Carbon Monoxide", min_value=0.0, value=1.0)
                nmhc_gt = st.number_input("NMHC(GT) - Non-Methane Hydrocarbons", min_value=0.0, value=1.0)
                c6h6_gt = st.number_input("C6H6(GT) - Benzene", min_value=0.0, value=1.0)
            with col2:
                nox_gt = st.number_input("NOx(GT) - Nitrogen Oxides", min_value=0.0, value=1.0)
                no2_gt = st.number_input("NO2(GT) - Nitrogen Dioxide", min_value=0.0, value=1.0)
                pt08_o3 = st.number_input("PT08.S5(O3) - Ozone Sensor", min_value=0.0, value=1.0)
            with col3:
                temp = st.number_input("T - Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)
                humidity = st.number_input("RH - Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
                absolute_humidity = st.number_input("AH - Absolute Humidity", min_value=0.0, value=5.0)

            input_values = {
                'CO(GT)': co_gt,
                'NMHC(GT)': nmhc_gt,
                'C6H6(GT)': c6h6_gt,
                'NOx(GT)': nox_gt,
                'NO2(GT)': no2_gt,
                'PT08.S5(O3)': pt08_o3,
                'T': temp,
                'RH': humidity,
                'AH': absolute_humidity
            }

            if st.button("Predict Air Quality Index"):
                    predicted_aqi = predict_aqi(input_values, nn_model, xgb_model, rf_model, scaler)
                    st.session_state.aqi_predicted = True
                    st.session_state.predicted_aqi = predicted_aqi

    elif option == "Manually Enter AQI Value":
        aqi_value = st.number_input("Enter the AQI value", min_value=0.0, step=0.01)
        if st.button("Submit Air Quality Index"):
            st.session_state.predicted_aqi = aqi_value    
            st.session_state.aqi_predicted = True

    if st.session_state.aqi_predicted:
        # Display AQI result
        st.subheader("Predicted AQI")
        predicted_aqi = st.session_state.predicted_aqi
        st.metric("AQI Reading", f"{predicted_aqi:.2f}")

        if predicted_aqi <= 50:
            st.success("Air Quality: Good")
        elif predicted_aqi <= 100:
            st.warning("Air Quality: Moderate")
        elif predicted_aqi <=150:
            st.warning("Air Quality: Unhealthy for Sensitive Groups")
        elif predicted_aqi <= 200:
            st.warning("Air Quality: Unhealthy")
        elif predicted_aqi <= 300:
            st.error("Air Quality: Very Unhealthy")
        elif predicted_aqi >= 300:
            st.error("Air Quality: Hazardous")
        else:
            st.error("Invalid AQI value")

    # Post-recommendation interaction
    if st.session_state.aqi_predicted:
        with st.expander("Provide Insights for Recommendations", expanded=st.session_state.aqi_predicted):
            st.session_state.answers = {}
            st.session_state.answers['traffic'] = st.selectbox("Traffic Intensity", ["Very High", "Moderate", "Low"])
            st.session_state.answers['industrial'] = st.selectbox("Proximity to Industrial Zones", ["Within 5 km", "Within 10 km", "More than 10 km away"])
            st.session_state.answers['green_cover'] = st.selectbox("Green Cover", ["Sparse", "Moderate", "Dense"])
            st.session_state.answers['climate_initiatives'] = st.selectbox("Climate Change Initiatives", ["None", "A few", "Actively implemented"])
            st.session_state.answers['zoning'] = st.radio("Mixed Zoning (Residential + Commercial)", ["Yes", "No"])
            st.session_state.answers['environmental_challenge'] = st.text_area("Significant Environmental Challenge", key="environmental_challenge", placeholder="For example, too much dust due to nearby construction or heavy traffic on main roads.")
            st.session_state.answers['area_initiatives'] = st.text_area("Recent Initiatives or Unique Aspects", key="area_initiatives", placeholder="For example, a new metro station is being built nearby.")

            if st.button("Get Recommendations"):
                if not st.session_state.answers['environmental_challenge'] or not st.session_state.answers['area_initiatives']:
                    st.error("Please fill in all required fields.")
                else:
                    prompt = construct_prompt(st.session_state.predicted_aqi, st.session_state.answers)
                    try:
                        response = model.generate_content(prompt)
                        st.write("### Recommendations")
                        st.write(response.text)
                        st.session_state.show_detailed_insights=True # Set this state to show the detailed insights button
                    except Exception as e:
                        st.error(f"Error generating insights: {e}")

        # Follow-up interaction
        if st.session_state.show_detailed_insights:
            st.subheader("Ask More About These Recommendations")
            follow_up_query = st.text_input("Have a follow-up question? Ask here:", key="follow_up_query")
            
            if st.button("Get Detailed Insights"):
                if follow_up_query.strip():
                    follow_up_prompt = construct_prompt(st.session_state.predicted_aqi, st.session_state.answers, follow_up_query)
                    try:
                        follow_up_response = model.generate_content(follow_up_prompt)
                        st.write("### Detailed Insights")
                        st.write(follow_up_response.text)
                    except Exception as e:
                        st.error(f"Error generating insights: {e}")
                else:
                    st.error("Please enter a valid question.")

if __name__ == "__main__":
    main()