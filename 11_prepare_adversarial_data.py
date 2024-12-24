import numpy as np
import pandas as pd

def prepare_adversarial_data():
    """
    הכנת הנתונים לאימון אדברסרי:
    1. טעינת הנתונים המקוריים
    2. חלוקה לסטים שונים של נתונים
    3. שמירת הנתונים המוכנים
    """
    # טעינת הנתונים המקוריים
    data = np.load('data/training_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    mask_train = data['mask_train']
    
    # יצירת סטים שונים של נתונים מופרעים לאימון
    noise_levels = [0.05, 0.1, 0.15]  # רמות שונות של רעש
    X_train_sets = []
    
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, X_train.shape)
        X_train_noisy = X_train + noise
        X_train_sets.append(X_train_noisy)
    
    # שמירת הנתונים המוכנים
    np.savez('data/adversarial_data.npz',
             X_train=X_train,                    # נתוני אימון מקוריים
             X_train_noisy_sets=X_train_sets,    # סטים של נתונים מופרעים
             noise_levels=noise_levels,          # רמות הרעש ששימשו
             y_train=y_train,
             mask_train=mask_train)
    
    print('Adversarial training data prepared and saved:')
    print(f'- Original training samples: {X_train.shape[0]}')
    print(f'- Noise levels prepared: {noise_levels}')
    print(f'- Total noisy sets: {len(X_train_sets)}')

if __name__ == "__main__":
    prepare_adversarial_data()
