import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

def evaluate_adversarial_model():
    """
    הערכת המודל האדברסרי:
    1. השוואה למודל הבסיסי
    2. בדיקת עמידות להפרעות
    3. ניתוח לפי קטגוריות
    """
    # טעינת נתוני הבדיקה
    data = np.load('data/training_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    mask_test = data['mask_test']
    
    # הגדרת התכונות
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # טעינת המודלים
    base_model = keras.models.load_model('models/finetuned_model.keras')
    adv_model = keras.models.load_model('models/adversarial_model.keras')
    
    # חיזוי
    base_predictions = base_model.predict(X_test)
    adv_predictions = adv_model.predict(X_test)
    
    # ניתוח לפי קטגוריות
    results = []
    for i, feature in enumerate(features):
        # מציאת האינדקסים של הערכים הממוסכים
        feature_mask = mask_test[:, i]
        
        if feature_mask.any():
            # חישוב מדדים למודל הבסיסי
            base_mae = np.mean(np.abs(y_test[feature_mask, i] - base_predictions[feature_mask, i]))
            base_mse = np.mean((y_test[feature_mask, i] - base_predictions[feature_mask, i])**2)
            base_rmse = np.sqrt(base_mse)
            
            # חישוב מדדים למודל האדברסרי
            adv_mae = np.mean(np.abs(y_test[feature_mask, i] - adv_predictions[feature_mask, i]))
            adv_mse = np.mean((y_test[feature_mask, i] - adv_predictions[feature_mask, i])**2)
            adv_rmse = np.sqrt(adv_mse)
            
            results.append({
                'Feature': feature,
                'Base_MAE': base_mae,
                'Base_RMSE': base_rmse,
                'Adv_MAE': adv_mae,
                'Adv_RMSE': adv_rmse,
                'MAE_Diff': base_mae - adv_mae,
                'RMSE_Diff': base_rmse - adv_rmse
            })
    
    # יצירת DataFrame
    results_df = pd.DataFrame(results)
    
    # שמירת התוצאות
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/feature_comparison.csv', index=False)
    
    # יצירת גרף השוואה
    plt.figure(figsize=(12, 6))
    x = range(len(features))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], results_df['Base_MAE'], width, 
            label='Base Model', color='blue', alpha=0.6)
    plt.bar([i + width/2 for i in x], results_df['Adv_MAE'], width, 
            label='Adversarial Model', color='red', alpha=0.6)
    
    plt.xlabel('Features')
    plt.ylabel('MAE')
    plt.title('Model Performance Comparison by Feature')
    plt.xticks(x, features, rotation=45)
    plt.legend()
    
    # שמירת הגרף
    os.makedirs('plots/adversarial', exist_ok=True)
    plt.savefig('plots/adversarial/feature_comparison.png', bbox_inches='tight')
    plt.close()
    
    print("\nResults saved to results/feature_comparison.csv")
    print("Plot saved to plots/adversarial/feature_comparison.png")
    
    return results_df

if __name__ == "__main__":
    results = evaluate_adversarial_model()
    print("\nFeature-wise comparison:")
    print(results)
