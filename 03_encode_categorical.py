# 03_encode_categorical.py
# This script encodes the categorical features
# It creates a new column 'AgeCategory' based on the 'HouseAge' column
# The purpose of this script is to convert categorical features to numerical features.
# This is a necessary step in the data preparation process for machine learning models, as most models can only work with numerical data.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Load the prepared data
df = pd.read_csv('data/prepared_data.csv')

# Encode categorical features
label_encoder = LabelEncoder()
df['AgeCategory'] = label_encoder.fit_transform(df['AgeCategory']) + 1

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

unique_values = df['AgeCategory'].unique()
print(f'Unique values in AgeCategory: {unique_values}')

# Save the label encoder for future use
import pickle
with open('models/label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# Save the encoded data for the next script
df.to_csv('data/encoded_data.csv', index=False)

print('Categorical features encoded and saved to data/encoded_data.csv')