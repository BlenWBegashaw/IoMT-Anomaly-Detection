import streamlit as st
import pandas as pd
import joblib
import time
import os
import numpy as np
import socket
import threading
import uvicorn
from api import app  # Import FastAPI app

# ✅ Load trained models and scaler
try:
    model_rf = joblib.load("model/random_forest.pkl")
    model_svm = joblib.load("model/svm_model.pkl")
    print("✅ RandomForest and SVM models loaded.")
except:
    raise ValueError("❌ Error: Models not found! Train them first.")

scaler = joblib.load("model/scaler.pkl")

# ✅ Load dataset
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

# ✅ Initialize empty DataFrames for tracking anomalies
live_df = pd.DataFrame(columns=df.columns)
anomalies_rf = pd.DataFrame(columns=df.columns)
anomalies_svm = pd.DataFrame(columns=df.columns)

# ✅ Function to simulate real-time detection
def detect_anomalies():
    global live_df, anomalies_rf, anomalies_svm
    live_df = pd.DataFrame(columns=df.columns)  # Reset live data
    anomalies_rf = pd.DataFrame(columns=df.columns)  # Reset anomaly tracking
    anomalies_svm = pd.DataFrame(columns=df.columns)

    for i in range(len(df)):
        row = df.iloc[i:i+1]  # Extract a single-row DataFrame
        scaled_row = scaler.transform(row)  # Scale it
        
        # Predict anomaly using both models
        pred_rf = model_rf.predict(scaled_row)[0]
        pred_svm = model_svm.predict(scaled_row)[0]
        
        # Determine anomaly status
        is_anomaly_rf = pred_rf == 1  # Assuming 1 = Anomaly
        is_anomaly_svm = pred_svm == 1

        # Append new row to live DataFrame
        if not live_df.empty:
            live_df = pd.concat([live_df, row])
        else:
            live_df = row.copy()

        # Track anomalies separately
        if is_anomaly_rf:
            anomalies_rf = pd.concat([anomalies_rf, row]) if not anomalies_rf.empty else row.copy()

        if is_anomaly_svm:
            anomalies_svm = pd.concat([anomalies_svm, row]) if not anomalies_svm.empty else row.copy()

        # Status message
        status_msg = "✅ Normal"
        if is_anomaly_rf or is_anomaly_svm:
            status_msg = f"⚠️ Anomaly Detected (RF: {is_anomaly_rf}, SVM: {is_anomaly_svm})"

        time.sleep(3)  # Simulate real-time processing
        yield live_df, anomalies_rf, anomalies_svm, status_msg

# ✅ Streamlit UI
st.title("IoMT Anomaly Detection Dashboard")

placeholder = st.empty()

# Streamlit loop to update data in real-time
for live_data, rf_anomalies, svm_anomalies, status in detect_anomalies():
    with placeholder.container():
        st.subheader("📊 Live IoMT Data")
        st.dataframe(live_data)

        st.subheader("🚨 RandomForest Detected Anomalies")
        st.dataframe(rf_anomalies)

        st.subheader("⚡ SVM Detected Anomalies")
        st.dataframe(svm_anomalies)

        st.subheader("Status")
        st.write(status)

        time.sleep(3)  # Simulate real-time update

# ✅ Run FastAPI alongside Streamlit
def run_fastapi():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()
