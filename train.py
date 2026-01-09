import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Paths ----------------
DATA_PATH = "dataset/winequality-red.csv"
MODEL_DIR = "output/model"
RESULT_DIR = "output/results"

# Create output directories automatically
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- Load dataset ----------------
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# ---------------- Preprocessing ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Train-test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- Model ----------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# REQUIRED: print metrics
print(f"MSE={mse}")
print(f"R2={r2}")

# ---------------- Save artifacts ----------------
joblib.dump(model, f"{MODEL_DIR}/model.pkl")

metrics = {
    "mse": mse,
    "r2_score": r2
}

with open(f"{RESULT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
