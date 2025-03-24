import pandas as pd
import time
import json

# Load dataset
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

print("🔵 IoMT Data Streaming Started...")

for index, row in df.iterrows():
    data = json.dumps(row.to_dict())  # Convert to JSON
    print(f"Streaming Data: {data}")  # Simulate sending data
    time.sleep(1)  # Simulate real-time delay
