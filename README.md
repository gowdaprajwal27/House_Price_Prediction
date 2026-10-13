# 🏠 House Price Prediction Using Machine Learning

## 📌 Project Overview
This project predicts median house values using Machine Learning
and the California Housing dataset.

A Linear Regression model is trained using housing-related features
such as median income, house age, average rooms, population,
and geographical coordinates.

## 🎯 Objectives
- Explore and analyze housing data.
- Preprocess the dataset.
- Train a Linear Regression model.
- Evaluate prediction performance.
- Predict house values using user-provided inputs.
- Save the trained model for future use.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 📂 Project Structure

House_Price_Prediction/
├── house_price_prediction.py
├── house_price_model.pkl
├── requirements.txt
└── README.md

## 📊 Dataset
The project uses the California Housing dataset provided by
Scikit-learn.

The dataset contains 20,640 records and 8 input features.
The target variable is MedHouseVal.

The target represents median house value in units of $100,000.

## 🤖 Machine Learning Model
Algorithm: Linear Regression

Dataset split:
- Training data: 80%
- Testing data: 20%

## 📈 Model Evaluation

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.5332 |
| Mean Squared Error (MSE) | 0.5559 |
| Root Mean Squared Error (RMSE) | 0.7456 |
| R² Score | 0.5758 |

These results are based on the current model run.

## 🚀 How to Run

1. Install Python.
2. Clone or download this repository.
3. Open the project folder in VS Code.
4. Install the required libraries:

   pip install -r requirements.txt

5. Run the program:

   python house_price_prediction.py

6. Close the visualization windows when they appear.
7. Enter the requested housing features to obtain a prediction.

## 💾 Saved Model
The trained model is saved as:

house_price_model.pkl

## ⚠️ Disclaimer
This project estimates median house values for California
districts. It does not predict the exact market price of an
individual house.

The model is intended for educational purposes.

## 👨‍💻 Author
Machine Learning Internship Project

## Running the Improved Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Predict house prices

```bash
python predict_house.py
```