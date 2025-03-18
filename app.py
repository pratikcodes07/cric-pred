import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model and encoders
data = joblib.load("model_and_encoders.pkl")
model = data["model"]
venue_encoder = data["venue_encoder"]
batting_team_encoder = data["batting_team_encoder"]
bowling_team_encoder = data["bowling_team_encoder"]
striker_encoder = data["striker_encoder"]
bowler_encoder = data["bowler_encoder"]

# Load dataset for dropdown values
df = pd.read_csv("ipl_data.csv")

# Streamlit UI
st.title("🏏 IPL Score Predictor")
st.markdown("Predict the final score of an innings based on match conditions!")

# Input fields
venue = st.selectbox("Select Venue", df['venue'].unique())
batting_team = st.selectbox("Select Batting Team", df['bat_team'].unique())
bowling_team = st.selectbox("Select Bowling Team", df['bowl_team'].unique())
striker = st.selectbox("Select Striker", df['batsman'].unique())
bowler = st.selectbox("Select Bowler", df['bowler'].unique())

runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, step=1)
wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, step=1)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=50.0, step=0.1)
runs = st.number_input("Current Runs", min_value=0, step=1)
wickets = st.number_input("Current Wickets", min_value=0, max_value=10, step=1)

# Prediction function
def predict_score():
    # Encode categorical inputs
    venue_encoded = venue_encoder.transform([venue])[0]
    batting_team_encoded = batting_team_encoder.transform([batting_team])[0]
    bowling_team_encoded = bowling_team_encoder.transform([bowling_team])[0]
    striker_encoded = striker_encoder.transform([striker])[0]
    bowler_encoded = bowler_encoder.transform([bowler])[0]

    # Feature engineering
    runs_last_5_ratio = runs_last_5 / runs if runs else 0
    wickets_last_5_ratio = wickets_last_5 / wickets if wickets else 0

    # Prepare input array
    input_features = np.array([
        venue_encoded, batting_team_encoded, bowling_team_encoded, striker_encoded, bowler_encoded,
        runs_last_5, wickets_last_5, overs, runs, wickets, runs_last_5_ratio, wickets_last_5_ratio
    ]).reshape(1, -1)

    # Predict
    predicted_score = model.predict(input_features)[0]
    return int(predicted_score)

# Prediction button
if st.button("Predict Score"):
    predicted_score = predict_score()
    st.success(f"🏏 Predicted Final Score: **{predicted_score}**")
