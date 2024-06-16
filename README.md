# Traffic Volume Forecasting

This repository contains a Time Series Forecasting Model for predicting traffic volume at a specific junction using the ARIMA model.

## Overview

The goal of this project is to predict future traffic volumes (number of vehicles) at a specific junction based on historical traffic data. The data is processed, split into training and test sets, and used to train an ARIMA model. The trained model is then used to make predictions, which are evaluated and visualized.

## Dataset

The dataset (`traffic.csv`) contains the following columns:
- `DateTime`: The date and time of the traffic observation.
- `Junction`: The junction ID where the observation was made.
- `Vehicles`: The number of vehicles counted at that junction and time.
- `ID`: A unique identifier for each observation.

## Steps to Run the Project

### Step 1: Preprocess the Data

1. Load the data.
2. Convert the 'DateTime' column to datetime format.
3. Set 'DateTime' as the index.
4. Filter data for Junction 1.
5. Resample the data to hourly intervals.

### Step 2: Train-Test Split

1. Split the data into training (data until 2017) and test sets (data from 2018 onwards).

### Step 3: Model Selection and Training

1. Choose the ARIMA model for time series forecasting.
2. Train the ARIMA model using the training data.

### Step 4: Model Evaluation

1. Verify the data split.
2. Make predictions using the trained model.
3. Plot and compare the actual vs predicted values.

### Step 5: Save the Model

1. Save the trained model using `joblib`.

## Requirements

- pandas
- statsmodels
- matplotlib
- joblib

Install the required libraries using:
```bash
pip install pandas statsmodels matplotlib joblib
