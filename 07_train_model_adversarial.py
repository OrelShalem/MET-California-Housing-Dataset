import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
from keras.models import load_model
from keras.losses import Loss
from keras import backend as K

# ייבוא הקלאס AdversarialLoss מהקובץ המקורי
from build_model_adversarial import AdversarialLoss

# טעינת הנתונים
data = np.load('data/training_data.npz')
X_num, X_cat = data['X_num'], data['X_cat']
y_num, y_cat = data['y_num'], data['y_cat']

# טעינת המודל האדברסרי
model = load_model('models/self_supervised_model_adversarial.h5', 
                  custom_objects={'AdversarialLoss': AdversarialLoss})

# אימון המודל
history = model.fit(
    {'numerical_inputs': X_num, 'categorical_inputs': X_cat},
    {'numerical_outputs': y_num, 'categorical_outputs': y_cat},
    epochs=10,
    batch_size=32
)

# שמירת המודל המאומן
model.save('models/self_supervised_model_adversarial_trained.h5')

# הדפסת סטטיסטיקות
print("\nNormalization statistics:")
print("\nNumerical features:")
print(f"Mean: {np.mean(X_num, axis=0)}")
print(f"Standard deviation: {np.std(X_num, axis=0)}")
print(f"Minimum: {np.min(X_num, axis=0)}")
print(f"Maximum: {np.max(X_num, axis=0)}")

print("\nTarget values:")
print(f"Mean: {np.mean(y_num, axis=0)}")
print(f"Standard deviation: {np.std(y_num, axis=0)}")
print(f"Minimum: {np.min(y_num, axis=0)}")
print(f"Maximum: {np.max(y_num, axis=0)}") 