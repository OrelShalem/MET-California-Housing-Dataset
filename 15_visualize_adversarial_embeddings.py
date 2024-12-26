# 15_visualize_adversarial_embeddings.py
"""
This script visualizes the adversarial embeddings.
"""
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def analyze_adversarial_embeddings():
    """
    Analysis and visualization of the fine-tuned adversarial model:
    1. Comparison between the regular model and the adversarial model
    2. Check robustness to perturbations
    3. Visualization of the results
    """
    # Create directories
    os.makedirs('plots/adversarial_analysis', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load test data
    data = np.load('data/training_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    mask_test = data['mask_test']
    
    # Load models
    regular_model = keras.models.load_model('models/best_model_run_1.keras')
    adv_model = keras.models.load_model('models/finetuned_adversarial_model.keras')
    
    # Define features
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # Check performance with different noise levels
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for noise in noise_levels:
        X_noisy = X_test + np.random.normal(0, noise, X_test.shape)
        
        # Prediction with both models - now with two inputs
        regular_pred = regular_model.predict([X_noisy, mask_test])
        adv_pred = adv_model.predict([X_noisy, mask_test])
        
        # Calculate metrics for each feature
        for i, feature in enumerate(features):
            feature_mask = mask_test[:, i]
            
            # Calculate MAE for each model
            regular_mae = np.mean(np.abs(y_test[feature_mask, i] - regular_pred[feature_mask, i]))
            adv_mae = np.mean(np.abs(y_test[feature_mask, i] - adv_pred[feature_mask, i]))
            
            results.append({
                'Feature': feature,
                'Noise_Level': noise,
                'Regular_MAE': regular_mae,
                'Adversarial_MAE': adv_mae
            })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Create comparison plot for each feature
    for feature in features:
        feature_data = results_df[results_df['Feature'] == feature]
        
        plt.figure(figsize=(10, 6))
        plt.plot(feature_data['Noise_Level'], feature_data['Regular_MAE'], 
                 'b-', label='Regular Model')
        plt.plot(feature_data['Noise_Level'], feature_data['Adversarial_MAE'], 
                 'r-', label='Adversarial Model')
        plt.xlabel('Noise Level')
        plt.ylabel('MAE')
        plt.title(f'{feature} - Model Performance vs. Noise Level')
        plt.legend()
        plt.savefig(f'plots/adversarial_analysis/{feature}_comparison.png')
        plt.close()
                
    # Save results
    results_df.to_csv('results/adversarial_embeddings_analysis.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    results = analyze_adversarial_embeddings()
