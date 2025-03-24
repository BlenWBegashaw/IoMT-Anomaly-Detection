import paho.mqtt.client as mqtt
import joblib
import json
import numpy as np

# Load trained model and scaler
try:
    model = joblib.load("model/random_forest.pkl")
    print("✅ RandomForest model loaded.")
except:
    model = joblib.load("model/one_class_svm.pkl")
    print("⚠️ One-Class SVM model loaded (no labels found).")

scaler = joblib.load("model/scaler.pkl")

# MQTT setup
broker = "test.mosquitto.org"
topic = "iomt/security"

def on_message(client, userdata, msg):
    try:
        # Parse received message
        data = json.loads(msg.payload.decode())
        values = np.array(list(data.values())).reshape(1, -1)  # Convert to NumPy array

        # Scale data
        scaled_point = scaler.transform(values)

        # Predict anomaly
        prediction = model.predict(scaled_point)
        is_anomaly = prediction[0] == -1 if 'label' not in data else prediction[0] == 1

        # Print result with alert
        status = "⚠️ Anomaly Detected!" if is_anomaly else "✅ Normal Operation"
        print(f"Received Data: {data} - {status}")

    except Exception as e:
        print(f"Error processing message: {e}")

client = mqtt.Client()
client.on_message = on_message

client.connect(broker)
client.subscribe(topic)

print("🔵 IoMT MQTT Subscriber Listening for Data...")
client.loop_forever()
