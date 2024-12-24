import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def analyze_adversarial_embeddings():
    """
    ניתוח וויזואליזציה של המודל האדברסרי המשופר:
    1. השוואה בין המודל הרגיל למודל האדברסרי
    2. בדיקת עמידות להפרעות שונות
    3. ויזואליזציה של התוצאות
    """
    # יצירת תיקיות
    os.makedirs('plots/adversarial_analysis', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # טעינת נתוני הבדיקה
    data = np.load('data/training_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    mask_test = data['mask_test']
    
    # טעינת המודלים
    regular_model = keras.models.load_model('models/finetuned_model.keras')
    adv_model = keras.models.load_model('models/finetuned_adversarial_model.keras')
    
    # הגדרת התכונות
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # בדיקת ביצועים עם רמות רעש שונות
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for noise in noise_levels:
        X_noisy = X_test + np.random.normal(0, noise, X_test.shape)
        
        # חיזוי עם שני המודלים
        regular_pred = regular_model.predict(X_noisy)
        adv_pred = adv_model.predict(X_noisy)
        
        # חישוב מדדים לכל תכונה
        for i, feature in enumerate(features):
            feature_mask = mask_test[:, i]
            
            # חישוב MAE עבור כל מודל
            regular_mae = np.mean(np.abs(y_test[feature_mask, i] - regular_pred[feature_mask, i]))
            adv_mae = np.mean(np.abs(y_test[feature_mask, i] - adv_pred[feature_mask, i]))
            
            results.append({
                'Feature': feature,
                'Noise_Level': noise,
                'Regular_MAE': regular_mae,
                'Adversarial_MAE': adv_mae
            })
    
    # יצירת DataFrame עם התוצאות
    results_df = pd.DataFrame(results)
    
    # יצירת גרף השוואה לכל תכונה
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
    
    # שמירת התוצאות
    results_df.to_csv('results/adversarial_embeddings_analysis.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    results = analyze_adversarial_embeddings()
