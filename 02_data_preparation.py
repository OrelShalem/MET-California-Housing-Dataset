# 02_data_preparation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data
data = pd.read_csv('data/raw_data.csv')
df = data.copy()

# Create age category by labeling the bins
df['AgeCategory'] = pd.cut(
    df['HouseAge'],
    bins=[0, 10, 30, 52],
    labels=['New', 'Mid', 'Old']
)

# Drop the original 'HouseAge' column
df = df.drop(columns=['HouseAge'])

# Define the numerical features
numerical_features = [
    'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
    'AveOccup', 'Latitude', 'Longitude'
]

# Define the categorical feature
categorical_features = ['AgeCategory']

# Normalize the features
scaler_features = StandardScaler()
scaler_target = StandardScaler()

# Fit the scalers and normalize the features
df[numerical_features] = scaler_features.fit_transform(df[numerical_features])
# Important! Fit the scaler_target on the original data
scaler_target.fit(df[numerical_features])  # Only fit, not transform

# Save the scalers
joblib.dump(scaler_features, 'models/scaler_features.pkl')
joblib.dump(scaler_target, 'models/scaler_target.pkl')

# Save the data
df.to_csv('data/prepared_data.csv', index=False)
print('Data prepared and saved to data/prepared_data.csv')