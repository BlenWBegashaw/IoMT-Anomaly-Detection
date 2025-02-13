import pandas as pd

# Load dataset
train_df = pd.read_csv("data/IP-Based Flows Pre-Processed Train.csv")
test_df = pd.read_csv("data/IP-Based Flows Pre-Processed Test.csv")

# Show dataset structure
print("Train Data Info:")
print(train_df.info())
print("\nTest Data Info:")
print(test_df.info())

# Check first few rows
print(train_df.head())
