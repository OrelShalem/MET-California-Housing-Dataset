# 11_prepare_adversarial_data.py
"""
Preparing adversarial data:
1. Loading original data
2. Creating different sets of noisy data
3. Saving prepared data
"""
import numpy as np
import pandas as pd

def prepare_adversarial_data():
    """
    Preparing adversarial data for training:
    1. Load original data
    2. Create different sets of noisy data
    3. Save prepared data
    """
    # Load original data
    data = np.load('data/training_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    mask_train = data['mask_train']
    
    # Create different sets of noisy data for training
    noise_levels = [0.05, 0.1, 0.15]  # Different levels of noise 5%, 10%, 15%
    X_train_sets = []
    
    """
    # Create Gaussian noise for each level
    # Add noise to original data
    # Save each perturbed version
    """
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, X_train.shape)
        X_train_noisy = X_train + noise
        X_train_sets.append(X_train_noisy)
    
    # Save prepared data
    np.savez('data/adversarial_data.npz',
             X_train=X_train,                    # Original training samples
             X_train_noisy_sets=X_train_sets,    # Sets of perturbed data
             noise_levels=noise_levels,          # Noise levels used
             y_train=y_train,
             mask_train=mask_train)
    
    print('Adversarial training data prepared and saved:')
    print(f'- Original training samples: {X_train.shape[0]}')
    print(f'- Noise levels prepared: {noise_levels}')
    print(f'- Total noisy sets: {len(X_train_sets)}')

if __name__ == "__main__":
    prepare_adversarial_data()
