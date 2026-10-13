# House Price Prediction - Model Training
# Project 3: Machine Learning Internship

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "house_price_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load the dataset
print("\nLoading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Dataset shape: {df.shape}")

# 2. Check and handle missing values
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# Fill missing numeric values with their column medians.
# The current California Housing dataset has no missing values.
df = df.fillna(df.median(numeric_only=True))

# 3. Separate features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 5. Train the model
print("\nTraining Linear Regression model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed!")

# 6. Generate predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE:  ${mae * 100000:,.2f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: ${rmse * 100000:,.2f}")
print(f"R2 Score: {r2:.4f}")
print("======================================")

# 8. Save the trained model and feature names
model_bundle = {
    "model": model,
    "features": list(X.columns),
    "target": "MedHouseVal",
    "target_scale": 100000
}

joblib.dump(model_bundle, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH.name}")

# 9. Save the house price distribution graph
plt.figure(figsize=(9, 5))

sns.histplot(
    df["MedHouseVal"],
    bins=50,
    kde=True
)

plt.title("Distribution of California House Values")
plt.xlabel("Median House Value ($100,000s)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=150)
plt.close()

# 10. Save the correlation heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

# 11. Save the actual vs predicted graph
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.35)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual House Value ($100,000s)")
plt.ylabel("Predicted House Value ($100,000s)")
plt.title("Actual vs Predicted House Values")

plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
plt.close()

print("\nVisualizations saved in the outputs folder.")
print("Training pipeline completed successfully!")