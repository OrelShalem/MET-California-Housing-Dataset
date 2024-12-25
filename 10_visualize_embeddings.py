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
    
    # ביצוע חיזוי - עכשיו עם שני קלטים
    predictions = model.predict([X_test, mask_test])
    
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
        
        if np.any(feature_mask):  # בדיקה שיש ערכים ממוסכים
            # חישוב מדדי דיוק רק על הערכים הממוסכים
            masked_true = y_test[feature_mask == 1, i]
            masked_pred = predictions[feature_mask == 1, i]
            
            # חישוב מדדי דיוק
            mae = np.mean(np.abs(masked_true - masked_pred))
            mse = np.mean((masked_true - masked_pred)**2)
            rmse = np.sqrt(mse)
            
            # יצירת scatter plot
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
    
    # יצירת DataFrame עם התוצאות
    results_df = pd.DataFrame(results)
    
    # הדפסת סיכום התוצאות
    print("\nFine-tuned Model Results Summary:")
    print(results_df.to_string(index=False))
    
    # יצירת heatmap של מדדי הדיוק
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
    
    # שמירת התוצאות לקובץ
    results_df.to_csv('results/finetuned_metrics.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    results = analyze_finetuned_results()