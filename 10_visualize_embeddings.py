# 10_visualize_embeddings.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow import keras

def analyze_finetuned_results():
    """
    ניתוח תוצאות המודל המשופר:
    1. טעינת המודל המשופר
    2. ביצוע חיזוי
    3. השוואה לערכים המקוריים
    4. ויזואליזציה של התוצאות
    """
    # יצירת תיקיות
    os.makedirs('plots/analysis', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # טעינת הנתונים
    data = np.load('data/training_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    mask_test = data['mask_test']
    
    # טעינת המודל המשופר
    model = keras.models.load_model('models/finetuned_model.keras')
    
    # ביצוע חיזוי
    predictions = model.predict(X_test)
    
    # הגדרת התכונות
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # יישוב מדדי דיוק לכל תכונה
    results = []
    for i, feature in enumerate(features):
        # מציאת האינדקסים של הערכים הממוסכים
        feature_mask = mask_test[:, i]
        
        if feature_mask.any():
            # חישוב מדדי דיוק
            mae = np.mean(np.abs(y_test[feature_mask, i] - predictions[feature_mask, i]))
            mse = np.mean((y_test[feature_mask, i] - predictions[feature_mask, i])**2)
            rmse = np.sqrt(mse)
            
            # יצירת scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test[feature_mask, i], predictions[feature_mask, i], alpha=0.5)
            plt.plot([y_test[feature_mask, i].min(), y_test[feature_mask, i].max()], 
                     [y_test[feature_mask, i].min(), y_test[feature_mask, i].max()], 
                     'r--')
            plt.xlabel('Original Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{feature} - Original vs Predicted')
            plt.savefig(f'plots/analysis/{feature}_comparison.png')
            plt.close()
            
            results.append({
                'Feature': feature,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'Predicted_Count': feature_mask.sum()
            })
    
    # יצירת DataFrame עם התוצאות
    results_df = pd.DataFrame(results)
    
    # הדפסת סיכום התוצאות
    print("\nFine-tuned Model Results Summary:")
    print(results_df.to_string(index=False))
    
    # יצירת heatmap של מדדי הדיוק
    plt.figure(figsize=(10, 6))
    sns.heatmap(results_df[['MAE', 'MSE', 'RMSE']].values.reshape(1, -1),
                annot=True, fmt='.4f',
                xticklabels=['MAE', 'MSE', 'RMSE'],
                yticklabels=['Metrics'])
    plt.title('Fine-tuned Model Error Metrics')
    plt.savefig('plots/analysis/error_metrics_heatmap.png')
    plt.close()
    
    # שמירת התוצאות לקובץ
    results_df.to_csv('results/finetuned_metrics.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    results = analyze_finetuned_results()