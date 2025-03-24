import pandas as pd
import joblib
import time

# Load trained model and scaler
try:
    model = joblib.load("model/random_forest.pkl")
    print("✅ RandomForest model loaded.")
except:
    model = joblib.load("model/one_class_svm.pkl")
    print("⚠️ One-Class SVM model loaded (no labels found).")

scaler = joblib.load("model/scaler.pkl")

# Load dataset (simulate real-time)
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

print("🔴 Real-Time IoMT Threat Detection Started...")

# Simulate live data processing
for index, row in df.iterrows():
    data_point = row.values.reshape(1, -1)  # Convert row to 2D array
    scaled_point = scaler.transform(data_point)  # Scale the data

    # Predict anomaly
    prediction = model.predict(scaled_point)
    is_anomaly = prediction[0] == -1 if 'label' not in df.columns else prediction[0] == 1

    # Print result with alert
    status = "⚠️ Anomaly Detected!" if is_anomaly else "✅ Normal Operation"
    print(f"New Data: {row.to_dict()} - {status}")

    time.sleep(1)  # Simulate real-time processing
