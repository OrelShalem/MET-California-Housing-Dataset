# 02_data_preparation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# טעינת הנתונים
data = pd.read_csv('data/raw_data.csv')
df = data.copy()

# יצירת קטגוריית גיל
df['AgeCategory'] = pd.cut(
    df['HouseAge'],
    bins=[0, 10, 30, 52],
    labels=['New', 'Mid', 'Old']
)
df = df.drop(columns=['HouseAge'])

numerical_features = [
    'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
    'AveOccup', 'Latitude', 'Longitude'
]
categorical_features = ['AgeCategory']

# נרמול המאפיינים
scaler_features = StandardScaler()
scaler_target = StandardScaler()

# התאמת הסקיילרים והנרמול
df[numerical_features] = scaler_features.fit_transform(df[numerical_features])
# חשוב! להתאים את scaler_target על הנתונים המקוריים
scaler_target.fit(df[numerical_features])  # רק fit, לא transform

# שמירת הסקיילרים
joblib.dump(scaler_features, 'models/scaler_features.pkl')
joblib.dump(scaler_target, 'models/scaler_target.pkl')

# שמירת הנתונים
df.to_csv('data/prepared_data.csv', index=False)
print('Data prepared and saved to data/prepared_data.csv')