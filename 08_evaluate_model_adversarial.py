import numpy as np
from tensorflow import keras
from build_model_adversarial import AdversarialLoss

# טעינת הנתונים
data = np.load('data/training_data.npz')
X_num, X_cat = data['X_num'], data['X_cat']
y_num, y_cat = data['y_num'], data['y_cat']

# טעינת המודל האדברסרי המאומן
model = keras.models.load_model('models/self_supervised_model_adversarial_trained.h5',
                              custom_objects={'AdversarialLoss': AdversarialLoss})

# הערכת המודל
predictions = model.predict([X_num, X_cat])
y_num_pred, y_cat_pred = predictions

# חישוב מדדי ביצוע
mse = np.mean((y_num - y_num_pred) ** 2)
print(f"Mean Squared Error for numerical features: {mse}")

# חישוב דיוק לתכונות הקטגוריות
cat_accuracy = np.mean(np.argmax(y_cat_pred, axis=1) == y_cat.ravel())
print(f"Accuracy for categorical feature: {cat_accuracy}")

# MSE לכל מאפיין מספרי
feature_names = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 
                'AveOccup', 'Latitude', 'Longitude']
for i, name in enumerate(feature_names):
    feature_mse = np.mean((y_num[:, i] - y_num_pred[:, i]) ** 2)
    print(f"MSE for {name}: {feature_mse}")

print("Evaluation completed.") 