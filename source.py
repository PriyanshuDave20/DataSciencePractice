'''
import os
import math
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from hpelm import ELM
import keras as k
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.regularizers import l2
import yfinance as yf
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# List of ticker symbols
tickers = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT", "META"]

# Download historical stock data
dataTotal = yf.download(tickers, start="2012-01-01", end="2024-01-01")

# Drop NaN and duplicate rows
dataTotal.dropna(inplace=True)
dataTotal.drop_duplicates(inplace=True)

# Reset index to make Date a column
dataTotal.reset_index(inplace=True)

# Ensure numeric conversion for Volume and handle errors
for ticker in tickers:
    dataTotal["Volume", ticker] = pd.to_numeric(dataTotal["Volume", ticker], errors='coerce')

# Select only numeric columns for z-score filtering
numeric_data = dataTotal.select_dtypes(include=[np.number])

# Filter outliers in rows based on z-score
dataTotal = dataTotal[(np.abs(stats.zscore(numeric_data)) < 3).all(axis=1)]

#Properly formats date as a column and reindexes all remaining rows
dataTotal.set_index("Date", inplace=True)
dataTotal.reset_index(inplace=True)

names = ['Linear Regression', 'Support Vector Regression', 'Feedforward Neural Network', 'Convoluted Neural Network', 'Random Forest Regression', 
'Gradient Boosting Regression', 'Elastic Net', 'K Nearest Neighbours', 'Extreme Learning Machine', 'Bayesian Ridge']
#names = ['Linear Regression', 'Support Vector Regression', 'Random Forest Regression', 
#'Gradient Boosting Regression', 'Elastic Net', 'K Nearest Neighbours', 'Bayesian Ridge',"Extreme Learning Machine"]
mse = [0]*len(names)
r2 = [0]*len(names)

for ticker in tickers:
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    data = dataTotal.loc[:,(['High','Open','Close','Low'], ticker)]

    X = data[["Open", "High", "Low"]]  #feature set
    y = data["Close"]  #target variable
    
    #Scaling x to ensure all features contribute equally to the model's performance
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    X_reshaped = X.values.reshape(-1,3,1) #feature set for CNN
    Y_reshaped = y.values.reshape(-1,1) #target variable for CNN
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X_reshaped, Y_reshaped, test_size=0.2, random_state=42)

    LigReg = LinearRegression()
    Svr = MultiOutputRegressor(SVR(kernel='rbf', C=10, gamma='scale'))
    
    SNN = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),  # First layer with 64 units
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),  # Second hidden layer with 32 units
        Dense(1)  # Output layer with 1 unit for regression
    ])
    SNN.compile(optimizer='adam', loss='mse')
    CNN = Sequential([
        Conv1D(filters=128, kernel_size = 2, activation='relu', input_shape=(X1_train.shape[1],1)),
        Conv1D(filters=64, kernel_size = 2, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    CNN.compile(optimizer='adam', loss='mse')
    
    rfReg = RandomForestRegressor(n_estimators=100, random_state=42)
    gbReg = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3))
    enReg = ElasticNet(alpha = 0.01, l1_ratio=1.0)
    kNN = KNeighborsRegressor(n_neighbors=2)
    elm = ELM(X_train.shape[1],y_train.shape[1])
    elm.add_neurons(10,'sigm')
    brReg = MultiOutputRegressor(BayesianRidge())

    models = [LigReg, Svr, SNN,CNN, gbReg, enReg, rfReg, kNN, elm, brReg]
    #models = [LigReg, Svr, gbReg, enReg, rfReg, kNN, brReg,elm]

    #Train the models
    LigReg.fit(X_train,y_train)
    Svr.fit(X_train, y_train)
    SNN.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2, verbose=0)  
    CNN.fit(X1_train, y1_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
    rfReg.fit(X_train, y_train)
    gbReg.fit(X_train,y_train)
    enReg.fit(X_train, y_train)
    kNN.fit(X_train,y_train)
    elm.train(X_train,y_train)
    brReg.fit(X_train, y_train)

    predictions = []
    for i in range(len(models)):
        if models[i] != 'CNN':
            predictions.append(models[i].predict(X_test))
        else:
            predictions.append(models[i].predict(X1_test))
      

    # Evaluate the models
    for i in range(len(models)):
        mse[i] += mean_squared_error(y_test, predictions[i])
        r2[i] += r2_score(y_test, predictions[i])
    
for i in range(len(names)):
    print(f'The average mean square error for {names[i]}: {mse[i]/len(names)}')
    print(f'The average r for {names[i]}: {math.sqrt(r2[i]/len(names))}')
'''

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
from hpelm import ELM
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
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

# Functions
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
        'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting Regression': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)),
        'Elastic Net': ElasticNet(alpha=0.01, l1_ratio=1.0),
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
