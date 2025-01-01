"""
Stock Price Prediction and Evaluation Script
============================================

This script performs stock price prediction using various machine learning models and neural networks. 
It downloads historical stock data, preprocesses it, trains models, and evaluates their performance. 
The results are summarized using mean squared error (MSE) and R² scores for each model.

Features:
---------
1. Data Download:
   - Fetches historical stock data for specified tickers using Yahoo Finance.

2. Data Preprocessing:
   - Cleans the dataset by removing NaN values, duplicates, and outliers (using z-score filtering).
   - Scales the features and target variables using MinMaxScaler for model optimization.

3. Model Training:
   - Implements and trains a variety of machine learning models:
     - Linear Regression
     - Support Vector Regression (SVR)
     - Feedforward Neural Network (FNN)
     - Convolutional Neural Network (CNN)
     - Random Forest Regression
     - Gradient Boosting Regression
     - Elastic Net Regression
     - k-Nearest Neighbors (k-NN)
     - Extreme Learning Machine (ELM)
     - Bayesian Ridge Regression

4. Model Evaluation:
   - Evaluates each model using the test dataset.
   - Computes MSE and R² scores for performance comparison.

5. Results Summary:
   - Aggregates results across multiple stocks and prints the average MSE and R² scores.

Dependencies:
-------------
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `hpelm`
- `tensorflow`
- `keras`
- `yfinance`
- `seaborn`
- `scipy`

Usage:
------
1. Define the `tickers` list with stock ticker symbols.
2. Run the script to download data, train models, and evaluate performance.
3. Review printed results for insights into model accuracy and reliability.

Note:
-----
- Ensure all required libraries are installed in your Python environment.
- Modify `start` and `end` dates for data retrieval as needed.
- CNN and FNN models require reshaped input data, which is handled automatically.

Author:
-------
Priyanshu Dave

Date:
-----
01/01/2025

"""

import os
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from hpelm import ELM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import yfinance as yf
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
tickers = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT", "META"]
start_date = "2012-01-01"
end_date = "2024-01-01"
names = [
    'Linear Regression', 'Support Vector Regression', 'Feedforward Neural Network', 
    'Convoluted Neural Network', 'Random Forest Regression', 'Gradient Boosting Regression', 
    'Elastic Net', 'K Nearest Neighbours', 'Extreme Learning Machine', 'Bayesian Ridge'
]

def preprocess_data():
    data = yf.download(tickers, start=start_date, end=end_date)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.reset_index(inplace=True)

    # Convert Volume to numeric
    for ticker in tickers:
        data["Volume", ticker] = pd.to_numeric(data["Volume", ticker], errors='coerce')

    # Remove outliers based on z-score
    numeric_data = data.select_dtypes(include=[np.number])
    data = data[(np.abs(stats.zscore(numeric_data)) < 3).all(axis=1)]

    data.set_index("Date", inplace=True)
    data.reset_index(inplace=True)

    return data

def initialize_models(input_dim):
    param_grid1 = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}
    param_grid2 = {
        'n_estimators': np.linspace(50, 1000, 20).astype(int),
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'subsample': [0.5, 0.7, 1.0],
        'max_features': [None, 'sqrt', 'log2']
    }
    param_grid3 = {
    'alpha': [0.01, 0.1, 1.0, 10, 100],
    'l1_ratio': [0, 0.1, 0.5, 0.9, 1],
    'max_iter': [1000, 5000, 10000],
    'tol': [1e-4, 1e-5, 1e-6],
    'fit_intercept': [True, False],
    'normalize': [True, False]
    }
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': MultiOutputRegressor(SVR(kernel='rbf', C=10, gamma='scale')),
        'Feedforward Neural Network': Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            *[Dense(32, activation='relu') for _ in range(6)],
            Dense(1)
        ]),
        'Convoluted Neural Network': Sequential([
            Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(input_dim, 1)),
            Conv1D(filters=64, kernel_size=2, activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ]),
        'Random Forest Regression': GridSearchCV(estimator=RandomForestRegressor(n_estimators=100, random_state=42), param_grid=param_grid1, cv=5),
        'Gradient Boosting Regression': MultiOutputRegressor(GridSearchCV(estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3), param_grid=param_grid2, cv=5)),
        'Elastic Net': GridSearchCV(estimator=ElasticNet(alpha=0.01, l1_ratio=1.0), param_grid=param_grid3, cv=5),
        'K Nearest Neighbours': KNeighborsRegressor(n_neighbors=2),
        'Extreme Learning Machine': ELM(input_dim, 1),
        'Bayesian Ridge': MultiOutputRegressor(BayesianRidge())
    }

    models['Extreme Learning Machine'].add_neurons(10, 'sigm')
    models['Feedforward Neural Network'].compile(optimizer='adam', loss='mse')
    models['Convoluted Neural Network'].compile(optimizer='adam', loss='mse')

    return models

def train_and_evaluate_models(data, models):
    mse_scores = {name: 0 for name in models.keys()}
    r2_scores = {name: 0 for name in models.keys()}

    for ticker in tickers:
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        data_ticker = data.loc[:,(['High', 'Open', 'Close', 'Low'], ticker)]
        X = data_ticker[["Open", "High", "Low"]]
        y = data_ticker["Close"]

        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        X_reshaped = X_scaled.reshape(-1, 3, 1)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        X1_train, X1_test, y1_train, y1_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, random_state=42)

        for name, model in models.items():
            if name in ['Feedforward Neural Network', 'Convoluted Neural Network']:
                if name == 'Feedforward Neural Network':
                    model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2, verbose=0)
                else:
                    model.fit(X1_train, y1_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
            elif name == 'Extreme Learning Machine':
                model.train(X_train,y_train)
            else:
                model.fit(X_train, y_train)

            predictions = model.predict(X1_test if name == 'Convoluted Neural Network' else X_test)
            mse_scores[name] += mean_squared_error(y_test, predictions)
            r2_scores[name] += r2_score(y_test, predictions)

    return mse_scores, r2_scores

def main():
    data = preprocess_data()
    models = initialize_models(3)  # 3 features: Open, High, Low

    mse_scores, r2_scores = train_and_evaluate_models(data, models)

    for name in names:
        avg_mse = mse_scores[name] / len(tickers)
        avg_r2 = math.sqrt(r2_scores[name] / len(tickers))
        print(f'The average mean square error for {name}: {avg_mse}')
        print(f'The average r for {name}: {avg_r2}')

if __name__ == "__main__":
    main()
