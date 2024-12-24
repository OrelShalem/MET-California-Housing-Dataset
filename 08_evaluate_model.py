# 08_evaluate_model.py
# File 08 checks:
# - How close the model guessed to the actual value (MSE)
# - How many times the model guessed the category correctly (Accuracy)

import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the testing data
data = np.load('data/training_data.npz')
X_num = data['X_num']
X_cat = data['X_cat']
y_num = data['y_num']
y_cat = data['y_cat']

# Load the trained model
from keras.models import load_model
model = load_model('models/self_supervised_model.h5')

# Make predictions
predictions = model.predict({'numerical_inputs': X_num, 'categorical_inputs': X_cat})

# Extract predictions
y_num_pred = predictions[0]
y_cat_pred_probs = predictions[1]
y_cat_pred = np.argmax(y_cat_pred_probs, axis=1)

# Calculate MSE for numerical features
mse = mean_squared_error(y_num, y_num_pred)
print(f'Mean Squared Error for numerical features: {mse}')

# Calculate accuracy for categorical features
accuracy = accuracy_score(y_cat, y_cat_pred)
print(f'Accuracy for categorical feature: {accuracy}')

# Optionally, calculate MSE per feature
numerical_features = [
    'MedInc',
    'AveRooms',
    'AveBedrms',
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude'
]
# Calculate MSE for each numerical feature
for idx, feature in enumerate(numerical_features):
    feature_mse = mean_squared_error(y_num[:, idx], y_num_pred[:, idx])
    print(f'MSE for {feature}: {feature_mse}')

print('Evaluation completed.')