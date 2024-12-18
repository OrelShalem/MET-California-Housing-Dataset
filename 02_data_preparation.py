# 02_data_preparation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the raw data from the previous script
data = pd.read_csv('data/raw_data.csv')

# Copy the data to avoid altering the original dataset
df = data.copy()

# Introduce a categorical feature by binning 'HouseAge'
df['AgeCategory'] = pd.cut(
    df['HouseAge'],
    bins=[0, 10, 30, 52],
    labels=['New', 'Mid', 'Old']
)

# Drop 'HouseAge' if desired
df = df.drop(columns=['HouseAge'])

# Save the processed data for the next script
df.to_csv('data/prepared_data.csv', index=False)

print('Data prepared and saved to data/prepared_data.csv')