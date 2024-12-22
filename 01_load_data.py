# 01_load_data.py

# Import necessary libraries
from sklearn.datasets import fetch_california_housing 
import pandas as pd

# Load the dataset
california = fetch_california_housing()

# Create a DataFrame
# explain: california.data is the data, california.feature_names is the feature names
data = pd.DataFrame(california.data, columns=california.feature_names)

# explain: california.target is the target variable
# target variable will be used to predict the median house value
data['MedHouseVal'] = california.target

# Save the data to a CSV file for use in the next script
data.to_csv('data/raw_data.csv', index=False)

print('Data loaded and saved to data/raw_data.csv')