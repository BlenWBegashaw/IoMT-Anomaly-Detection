import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

print("Loading dataset...")
df = pd.read_csv("data/IP-Based Flows Pre-Processed Train.csv")
print("Dataset loaded!")

# Encode categorical "traffic" column
print("Encoding 'traffic' column...")
le = LabelEncoder()
df["traffic"] = le.fit_transform(df["traffic"])
print("Encoding completed!")

# Define features & target
X = df.drop(columns=["is_attack"])
y = df["is_attack"]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split completed!")

print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling completed!")

#  **NEW: Reduce dataset size for SVM training** 
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)
print(f"Reduced dataset size for SVM training: {X_train_sample.shape[0]} samples")

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest training completed!")

# Train SVM model (with optimizations)
print("Training SVM model...")
svm_model = SVC(kernel='linear', probability=True, max_iter=1000)
svm_model.fit(X_train_sample, y_train_sample)  # Train on smaller dataset
print("SVM training completed!")

# Evaluate models
print("Evaluating models...")
rf_preds = rf_model.predict(X_test)
svm_preds = svm_model.predict(X_test)

print("Random Forest Results:")
print(classification_report(y_test, rf_preds))

print("SVM Results:")
print(classification_report(y_test, svm_preds))

print("Training and evaluation completed!")
