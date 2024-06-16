# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 07:36:22 2024

@author: yashi
"""
#Step 1: Preprocess the Data

import pandas as pd

# Load the data
traffic_data = pd.read_csv('traffic.csv')

# Convert the 'DateTime' column to datetime format
traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])

# Set 'DateTime' as the index
traffic_data.set_index('DateTime', inplace=True)

# Filter data for Junction 1
junction_1_data = traffic_data[traffic_data['Junction'] == 1]['Vehicles']

# Resample to hourly data (though it seems to already be hourly)
junction_1_data = junction_1_data.resample('H').sum()

print(junction_1_data.head())

#Step 2: Train-Test Split

# Split the data into training and test sets
train_data = junction_1_data[:'2017']
test_data = junction_1_data['2018':]

print(train_data.shape, test_data.shape)

#Step 3: Model Selection and Training

from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model
model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit()

print(model_fit.summary())

#Step 4: Model Evaluation

# Verify the data split
print("Train Data Range:", train_data.index.min(), "to", train_data.index.max())
print("Test Data Range:", test_data.index.min(), "to", test_data.index.max())

# If the date ranges seem correct, proceed with prediction
if not test_data.empty:
    # Make predictions
    predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Train')
    plt.plot(test_data, label='Test')
    plt.plot(test_data.index, predictions, label='Predicted')
    plt.legend()
    plt.show()
else:
    print("Test data is empty. Please check the data split logic.")

#Step 5: Save the Model

import joblib

# Save the model
joblib.dump(model_fit, 'traffic_arima_model.pkl')
