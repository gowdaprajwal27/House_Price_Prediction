# House Price Prediction - Prediction Script

import joblib
import pandas as pd
from pathlib import Path

# Locate the saved model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"

# Check whether the model exists
if not MODEL_PATH.exists():
    print("Model not found!")
    print("Please run train_model.py first.")
    raise SystemExit(1)

# Load the trained model bundle
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]
target_scale = bundle["target_scale"]

print("\n===================================")
print("       HOUSE PRICE PREDICTOR")
print("===================================")
print("Enter the housing details below.\n")

# Collect inputs in the correct feature order
values = []

for feature in features:
    value = float(input(f"Enter {feature}: "))
    values.append(value)

# Prepare input data
input_data = pd.DataFrame(
    [values],
    columns=features
)

# Predict house value
prediction = model.predict(input_data)[0]

# Convert the prediction into dollars
predicted_price = prediction * target_scale

print("\n===================================")
print(f"Predicted house value: ${predicted_price:,.2f}")
print("===================================")
print("\nNote: This estimates a California district's")
print("median house value, not an individual home's price.")