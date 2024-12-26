# 10_visualize_embeddings.py
"""
Visualization of embeddings:
1. Loading the model
2. Extracting embeddings
3. Visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow import keras

def analyze_finetuned_results():
    """
    Analysis of fine-tuned model results:
    1. Loading the fine-tuned model
    2. Making predictions
    3. Comparing to original values
    4. Visualizing the results
    """
    # Create directories
    os.makedirs('plots/analysis', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load data
    data = np.load('data/training_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    mask_test = data['mask_test']
    
    # Load fine-tuned model
    model = keras.models.load_model('models/finetuned_model.keras')
    
    # Make predictions - now with two inputs
    predictions = model.predict([X_test, mask_test])
    
    # Define features
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # Calculate metrics for each feature
    results = []
    for i, feature in enumerate(features):
        # Find indices of masked values
        feature_mask = mask_test[:, i]
        
        if np.any(feature_mask):  # Check if there are masked values
            # Calculate metrics only on masked values
            masked_true = y_test[feature_mask == 1, i]
            masked_pred = predictions[feature_mask == 1, i]
            
            # Calculate metrics
            mae = np.mean(np.abs(masked_true - masked_pred))
            mse = np.mean((masked_true - masked_pred)**2)
            rmse = np.sqrt(mse)
            
            # Create scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(masked_true, masked_pred, alpha=0.5)
            plt.plot([masked_true.min(), masked_true.max()], 
                     [masked_true.min(), masked_true.max()], 
                     'r--', label='Perfect Prediction')
            plt.xlabel('Original Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{feature} - Original vs Predicted\nMAE: {mae:.4f}, RMSE: {rmse:.4f}')
            plt.legend()
            plt.savefig(f'plots/analysis/{feature}_comparison.png')
            plt.close()
            
            results.append({
                'Feature': feature,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'Predicted_Count': np.sum(feature_mask)
            })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Print summary of results
    print("\nFine-tuned Model Results Summary:")
    print(results_df.to_string(index=False))
    
    # Create heatmap of metrics
    plt.figure(figsize=(12, 6))
    metrics_data = results_df[['MAE', 'MSE', 'RMSE']].values
    sns.heatmap(metrics_data, 
                annot=True, 
                fmt='.4f',
                xticklabels=['MAE', 'MSE', 'RMSE'],
                yticklabels=results_df['Feature'],
                cmap='YlOrRd')
    plt.title('Fine-tuned Model Error Metrics per Feature')
    plt.tight_layout()
    plt.savefig('plots/analysis/error_metrics_heatmap.png')
    plt.close()
    
    # Save results to file
    results_df.to_csv('results/finetuned_metrics.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    results = analyze_finetuned_results()