# 03_encode_categorical.py

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

# Save the label encoder for future use
import pickle
with open('models/label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# Save the encoded data for the next script
df.to_csv('data/encoded_data.csv', index=False)

print('Categorical features encoded and saved to data/encoded_data.csv')