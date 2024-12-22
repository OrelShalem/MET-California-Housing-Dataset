# 06_prepare_training_data.py

import pandas as pd
import numpy as np
import joblib

# טעינת הסקיילרים
scaler_features = joblib.load('models/scaler_features.pkl')
scaler_target = joblib.load('models/scaler_target.pkl')

# טעינת הנתונים
df_masked = pd.read_csv('data/masked_data.csv')
df_original = pd.read_csv('data/encoded_data.csv')

numerical_features = [
    'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
    'AveOccup', 'Latitude', 'Longitude'
]
categorical_features = ['AgeCategory']

# הכנת הקלט
X_num = df_masked[numerical_features].values
X_cat = df_masked[categorical_features].values

# החלפת NaN באפסים
X_num = np.nan_to_num(X_num, nan=0.0)

# הכנת ערכי המטרה
y_num = df_original[numerical_features].values

# נרמול פשוט של ערכי המטרה
y_num = scaler_target.transform(y_num)

y_cat = df_original[categorical_features].values

# שמירת הנתונים
np.savez('data/training_data.npz', 
         X_num=X_num, X_cat=X_cat, 
         y_num=y_num, y_cat=y_cat)

print('Training data prepared and saved to data/training_data.npz')