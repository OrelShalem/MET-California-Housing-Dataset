# 06_prepare_training_data.py

import pandas as pd
import numpy as np

# Load masked data
df_masked = pd.read_csv('data/masked_data.csv')

# Load original data for targets
df_original = pd.read_csv('data/encoded_data.csv')

# Prepare inputs
numerical_features = [
    'MedInc',
    'AveRooms',
    'AveBedrms',
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude'
]
categorical_features = ['AgeCategory']

X_num = df_masked[numerical_features].values
X_cat = df_masked[categorical_features].values

# Replace NaNs with zeros (since they are masked)
X_num = np.nan_to_num(X_num, nan=0)

# Prepare targets (original data)
y_num = df_original[numerical_features].values
y_cat = df_original[categorical_features].values

# Save inputs and targets for training
np.savez('data/training_data.npz', X_num=X_num, X_cat=X_cat, y_num=y_num, y_cat=y_cat)

print('Training data prepared and saved to data/training_data.npz')