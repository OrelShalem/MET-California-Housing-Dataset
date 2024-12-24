# 06_prepare_training_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_training_data(test_size=0.2):
    """
    הכנת הנתונים לפי גישת MET:
    1. נרמול פשוט של הנתונים
    2. חלוקה לאימון ובדיקה
    3. שמירת הנתונים המוכנים
    """
    # טעינת הנתונים
    df_masked = pd.read_csv('data/masked_data.csv')
    df_original = pd.read_csv('data/encoded_data.csv')
    
    # הגדרת התכונות
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # נרמול פשוט של כל הנתונים
    scaler = StandardScaler()
    
    # התאמת הscaler על הנתונים המקוריים (לא ממוסכים)
    scaler.fit(df_original[features])
    
    # הכנת הנתונים
    X = df_masked[features].values.copy()  # שימוש בcopy למניעת אזהרות
    y = df_original[features].values
    mask = (df_masked[features] != -1).values
    
    # נרמול הנתונים
    # נרמול כל הנתונים ושמירה על ערכי המיסוך
    X_normalized = scaler.transform(X)
    X_normalized[~mask] = -1  # החזרת ערכי המיסוך
    X = X_normalized
    
    # נרמול נתוני המטרה
    y = scaler.transform(y)
    
    # חלוקה לאימון ובדיקה
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        X, y, mask, test_size=test_size, random_state=42
    )
    
    # שמירת הנתונים
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
    
    # שמירת הscaler
    np.save('models/scaler.npy', {
        'mean_': scaler.mean_,
        'scale_': scaler.scale_
    })

if __name__ == "__main__":
    prepare_training_data()