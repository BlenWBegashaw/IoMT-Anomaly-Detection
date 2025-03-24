import pandas as pd

data = {}
with open("data/IoMT.csv", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        category = parts[0].strip()
        tokens = parts[1].strip().split()
        # Convert tokens to integers where possible
        tokens = [int(token) for token in tokens if token.isdigit()]
        data[category] = tokens

# Convert dictionary to DataFrame (each column is a token category)
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
df.to_csv("data/Training_parsed.csv", index=False)
print("Parsed data saved as data/Training_parsed.csv!")
