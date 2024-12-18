# 01_load_data.py

from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the dataset
california = fetch_california_housing()

# Create a DataFrame
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# Save the data to a CSV file for use in the next script
data.to_csv('data/raw_data.csv', index=False)

print('Data loaded and saved to data/raw_data.csv')