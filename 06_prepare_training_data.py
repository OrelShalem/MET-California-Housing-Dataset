# 06_prepare_training_data.py

import pandas as pd
import numpy as np
import joblib

# Load the scalers
scaler_features = joblib.load('models/scaler_features.pkl')
scaler_target = joblib.load('models/scaler_target.pkl')

# Load the data
df_masked = pd.read_csv('data/masked_data.csv')
df_original = pd.read_csv('data/encoded_data.csv')

numerical_features = [
    'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
    'AveOccup', 'Latitude', 'Longitude'
]
categorical_features = ['AgeCategory']

# Prepare the input
X_num = df_masked[numerical_features].values
X_cat = df_masked[categorical_features].values

# Replace NaN with 0.0
X_num = np.nan_to_num(X_num, nan=0.0)

# Prepare the target
y_num = df_original[numerical_features].values

# Simple normalization of the target
y_num = scaler_target.transform(y_num)

y_cat = df_original[categorical_features].values

# Save the data
np.savez('data/training_data.npz', 
         X_num=X_num, X_cat=X_cat, 
         y_num=y_num, y_cat=y_cat)

print('Training data prepared and saved to data/training_data.npz')