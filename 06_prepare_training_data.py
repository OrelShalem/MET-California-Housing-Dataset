# 06_prepare_training_data.py
# This script is designed to prepare the data for the training phase.
# It handles data normalization, splitting into training and test sets, and dealing with masked data.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_training_data(test_size=0.2):
    """
    Prepare the data for the training phase using the MET approach:
    1. Simple normalization of the data
    2. Split into training and test sets
    3. Save the prepared data
    """
    # Load the data
    df_masked = pd.read_csv('data/masked_data.csv')
    df_original = pd.read_csv('data/encoded_data.csv')

    # Define the features
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # Simple normalization of all the data
    scaler = StandardScaler()
    
    # Fit the scaler on the original data (not masked)
    scaler.fit(df_original[features])
    
    # Prepare the data
    X = df_masked[features].values.copy()  # Use copy to avoid warnings
    y = df_original[features].values
    mask = (df_masked[features] != -1).values
    
    # Normalize the data
    # Normalize all the data and keep the masked values
    X_normalized = scaler.transform(X)
    X_normalized[~mask] = -1  # Return the masked values
    X = X_normalized
    
    # Normalize the target data
    y = scaler.transform(y)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        X, y, mask, test_size=test_size, random_state=42
    )
    
    # Save the data
    np.savez('data/training_data.npz',
             X_train=X_train,
             X_test=X_test,
             y_train=y_train,
             y_test=y_test,
             mask_train=mask_train,
             mask_test=mask_test
    )
    
    print('Training data prepared and saved:')
    print(f'Training set shape: {X_train.shape}')
    print(f'Test set shape: {X_test.shape}')
    print(f'Masked values in training: {mask_train.sum()}')
    print(f'Masked values in test: {mask_test.sum()}')
    
    # Save the scaler
    np.save('models/scaler.npy', {
        'mean_': scaler.mean_,
        'scale_': scaler.scale_
    })

if __name__ == "__main__":
    prepare_training_data()