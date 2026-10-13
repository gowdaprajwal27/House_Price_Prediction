# Project 3: House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

# Load the dataset
print("Loading the housing dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Display the first five rows
print("\nFirst 5 Rows:")
print(df.head())

# Display the dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nDataset Columns:")
print(df.columns.tolist())

# Display information about the dataset
print("\nDataset Information:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Exploratory Data Analysis (EDA)

print("\nStatistical Summary:")
print(df.describe())

# Check the distribution of house prices
plt.figure(figsize=(8, 5))
sns.histplot(df["MedHouseVal"], bins=50, kde=True)

plt.title("Distribution of House Prices")
plt.xlabel("Median House Value ($100,000s)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))

correlation = df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nEDA completed successfully!")

# Step 3: Prepare Data and Train the Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Separate input features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("R² Score:", round(r2, 4))

# Step 4: Actual vs Predicted House Prices

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.4)

# Perfect prediction reference line
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual House Prices ($100,000s)")
plt.ylabel("Predicted House Prices ($100,000s)")
plt.title("Actual vs Predicted House Prices")

plt.legend()
plt.tight_layout()
plt.show()

print("\nActual vs Predicted graph generated successfully!")
# Step 5: Custom House Price Prediction

print("\n========== HOUSE PRICE PREDICTOR ==========")

# Get housing details from the user
print("\nEnter the following housing details:")

med_inc = float(input("Median income (dataset units): "))
house_age = float(input("House age (years): "))
ave_rooms = float(input("Average rooms per household: "))
ave_bedrms = float(input("Average bedrooms per household: "))
population = float(input("District population: "))
ave_occup = float(input("Average household occupancy: "))
latitude = float(input("Latitude: "))
longitude = float(input("Longitude: "))

# Prepare input in the same feature order used during training
user_data = pd.DataFrame(
    [[
        med_inc,
        house_age,
        ave_rooms,
        ave_bedrms,
        population,
        ave_occup,
        latitude,
        longitude
    ]],
    columns=X.columns
)

# Predict the house value
predicted_price = model.predict(user_data)[0]

# Convert from units of $100,000 to dollars
predicted_dollars = predicted_price * 100000

print("\n========== PREDICTION RESULT ==========")
print(f"Predicted house value: ${predicted_dollars:,.2f}")
print("=======================================")

# Step 6: Save the Trained Model

import joblib

joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully!")