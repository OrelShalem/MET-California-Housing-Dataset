# 04_apply_masking.py

# pandas is used to load the data
# numpy is used to mask the data
import pandas as pd
import numpy as np

# Load the encoded data
df = pd.read_csv('data/encoded_data.csv')

# Define the masking function
def mask_data(df, mask_fraction=0.70):
    # explain: df_masked is a copy of the dataframe
    # explain: mask_indices is the indices of the dataframe that are masked
    df_masked = df.copy()
    # explain: sample is used to randomly sample the dataframe
    # explain: frac is the fraction of the dataframe that is masked
    # explain: index is the indices of the dataframe that are masked
    mask_indices = df_masked.sample(frac=mask_fraction).index

    # Mask numerical features
    numerical_features = [
        'MedInc',
        'AveRooms',
        'AveBedrms',
        'Population',
        'AveOccup',
        'Latitude',
        'Longitude'
    ]
    # Mask numerical features
    for feature in numerical_features:
        df_masked.loc[mask_indices, feature] = np.nan # nan is the mask token for numerical features

    # Mask categorical features
    categorical_features = ['AgeCategory']
    for feature in categorical_features:
        df_masked.loc[mask_indices, feature] = 0 # 0 is the mask token for categorical features

    return df_masked, mask_indices

# Apply masking
df_masked, mask_indices = mask_data(df)

# Save the masked data and mask indices
df_masked.to_csv('data/masked_data.csv', index=False)
mask_indices.to_series().to_csv('data/mask_indices.csv', index=False)

print('Data masked and saved to data/masked_data.csv')