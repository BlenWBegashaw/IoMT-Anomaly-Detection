import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ✅ Load dataset
file_path = "data/IoMT_with_pseudo_labels.csv"  # Ensure this file has labels
df = pd.read_csv(file_path)

# ✅ Check and Assign Target Column
target_column = "pseudo_label"  # Change this if your dataset uses a different column name

if target_column not in df.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found in dataset! Available columns: {df.columns.tolist()}")

# ✅ Feature Scaling
scaler = joblib.load("model/scaler.pkl")
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Labels
X_scaled = scaler.transform(X)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Load trained models
rf_model = joblib.load("model/random_forest.pkl")
svm_model = joblib.load("model/svm_model.pkl")

# ✅ Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

# ✅ Evaluate Models
rf_metrics = evaluate_model(rf_model, X_test, y_test)
svm_metrics = evaluate_model(svm_model, X_test, y_test)

# ✅ Print Results
print("📊 Model Performance Comparison:")
print(f"🌲 RandomForest: {rf_metrics}")
print(f"⚡ SVM: {svm_metrics}")

# ✅ Save results to file
metrics_df = pd.DataFrame([rf_metrics, svm_metrics], index=["RandomForest", "SVM"])
metrics_df.to_csv("data/model_evaluation.csv", index=True)

print("✅ Evaluation results saved to 'results/model_evaluation.csv'")
