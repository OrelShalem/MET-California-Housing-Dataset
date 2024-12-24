import numpy as np
from tensorflow import keras
import pandas as pd

def predict_new_data(input_data_path):
    """
    חיזוי על נתונים חדשים לפי גישת MET:
    1. טעינת נתונים חדשים
    2. נרמול הנתונים
    3. חיזוי
    4. ביטול הנרמול
    """
    # טעינת הנתונים החדשים
    df_new = pd.read_csv(input_data_path)
    
    # טעינת המודל
    model = keras.models.load_model('models/trained_model.keras')
    
    # טעינת הscaler
    scaler_params = np.load('models/scaler.npy', allow_pickle=True).item()
    
    # שחזור הscaler
    class CustomScaler:
        def __init__(self, mean_, scale_):
            self.mean_ = mean_
            self.scale_ = scale_
            
        def transform(self, X):
            return (X - self.mean_) / self.scale_
            
        def inverse_transform(self, X):
            return X * self.scale_ + self.mean_
    
    scaler = CustomScaler(scaler_params['mean_'], scaler_params['scale_'])
    
    # הגדרת התכונות
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # נרמול הנתונים
    X = df_new[features].values
    X_normalized = scaler.transform(X)
    
    # חיזוי
    predictions = model.predict(X_normalized)
    
    # ביטול הנרמול
    predictions = scaler.inverse_transform(predictions)
    
    # יצירת DataFrame עם התוצאות
    results_df = pd.DataFrame(predictions, columns=features)
    
    # שמירת התוצאות
    output_path = 'results/predictions.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    return results_df

if __name__ == "__main__":
    # לדוגמה: חיזוי על נתונים חדשים
    input_path = 'data/new_data.csv'  # נתיב לקובץ עם נתונים חדשים
    predictions = predict_new_data(input_path)