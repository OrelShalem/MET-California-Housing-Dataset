# 08_evaluate_model.py
# File 08 checks:
# - How close the model guessed to the actual value (MSE)
# - How many times the model guessed the category correctly (Accuracy)

import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

def evaluate_model():
    """
    Model evaluation using MET:
    1. Testing performance on test set
    2. Calculation of different metrics
    3. Visualization of results
    """
    # Load data
    data = np.load('data/training_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    mask_test = data['mask_test']
    
    # Load model
    model = keras.models.load_model('models/best_model_run_1.keras')
    
    # Prediction - now with two inputs
    y_pred = model.predict([X_test, mask_test])

    # Calculate metrics only on masked values
    mse_per_feature = []
    mae_per_feature = []
    acc_per_feature = []
    
    for i in range(X_test.shape[1]):
        feature_mask = mask_test[:, i]
        mse = mean_squared_error(y_test[feature_mask, i], y_pred[feature_mask, i])
        mae = mean_absolute_error(y_test[feature_mask, i], y_pred[feature_mask, i])
        
        # Calculate accuracy for each feature
        correct = np.abs(y_test[feature_mask, i] - y_pred[feature_mask, i]) < 0.5
        acc = np.mean(correct)
        
        mse_per_feature.append(mse)
        mae_per_feature.append(mae)
        acc_per_feature.append(acc)
    
    # Create DataFrame with results
    features = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 
               'AveOccup', 'Latitude', 'Longitude', 'AgeCategory']
    
    results_df = pd.DataFrame({
        'Feature': features,
        'MSE': mse_per_feature,
        'MAE': mae_per_feature,
        'RMSE': np.sqrt(mse_per_feature),
        'Accuracy': acc_per_feature
    })
    
    # Print results
    print("\nModel Evaluation Results:")
    print(results_df.to_string(index=False))
    print(f"\nOverall MSE: {np.mean(mse_per_feature):.4f}")
    print(f"Overall MAE: {np.mean(mae_per_feature):.4f}")
    print(f"Overall RMSE: {np.sqrt(np.mean(mse_per_feature)):.4f}")
    print(f"Overall Accuracy: {np.mean(acc_per_feature):.4f}")
    
    # Create directories if they don't exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Visualization of results
    plt.figure(figsize=(15, 5))
    
    # Plot MSE per feature
    plt.subplot(1, 3, 1)
    plt.bar(features, mse_per_feature)
    plt.xticks(rotation=45)
    plt.title('MSE per Feature')
    
    # Plot MAE per feature
    plt.subplot(1, 3, 2)
    plt.bar(features, mae_per_feature)
    plt.xticks(rotation=45)
    plt.title('MAE per Feature')
    
    # Plot Accuracy per feature
    plt.subplot(1, 3, 3)
    plt.bar(features, acc_per_feature)
    plt.xticks(rotation=45)
    plt.title('Accuracy per Feature')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation_results.png')
    plt.close()
    
    # Save results to file
    results_df.to_csv('results/evaluation_metrics.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    results = evaluate_model()