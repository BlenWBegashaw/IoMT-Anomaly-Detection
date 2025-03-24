import pandas as pd
import joblib
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load dataset
file_path = "data/IoMT.csv"
df = pd.read_csv(file_path)

# Data Cleaning
df = df.drop_duplicates().fillna(0)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)  # Use all data for clustering

# **Step 1: Generate Pseudo-Labels using K-Means Clustering**
kmeans = KMeans(n_clusters=2, random_state=42)  # Assume 2 classes (normal/anomaly)
pseudo_labels = kmeans.fit_predict(X_scaled)  # Generate pseudo labels

# Assign pseudo-labels to DataFrame
df["pseudo_label"] = pseudo_labels


# ✅ Save the dataset with pseudo-labels
df.to_csv("data/IoMT_with_pseudo_labels.csv", index=False)
print("✅ Dataset with pseudo-labels saved as 'data/IoMT_with_pseudo_labels.csv'")


# **Step 2: Train RandomForest & SVM**
X = X_scaled
y = pseudo_labels  # Use pseudo-labels

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train SVM Model
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "model/random_forest.pkl")
joblib.dump(svm_model, "model/svm_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ RandomForest and SVM trained using pseudo-labels and saved in 'model/' folder.")
