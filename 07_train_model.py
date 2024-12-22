# 07_train_model.py

import tensorflow as tf
import numpy as np

# Load the training data
data = np.load('data/training_data.npz', allow_pickle=True)
X_num = data['X_num']
X_cat = data['X_cat']
y_num = data['y_num']
y_cat = data['y_cat']

# Verify that X_num and X_cat are scaled
print(f'X_num mean: {np.mean(X_num, axis=0)}')
print(f'X_num std: {np.std(X_num, axis=0)}')
print(f'y_num mean: {np.mean(y_num, axis=0)}')
print(f'y_num std: {np.std(y_num, axis=0)}')

# Print unique values in categorical inputs to verify the range
print(f'Categorical data range: {X_cat.min()} to {X_cat.max()}')
print(f'Unique values in X_cat: {np.unique(X_cat)}')

# Load the model architecture
from keras.models import model_from_json
with open('models/model_architecture.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Compile the model
model.compile(
    optimizer='adam',
    loss={
        'numerical_outputs': 'mse',
        'categorical_outputs': 'sparse_categorical_crossentropy'
    }
)

# Train the model
history = model.fit(
    {'numerical_inputs': X_num, 'categorical_inputs': X_cat},
    {'numerical_outputs': y_num, 'categorical_outputs': y_cat},
    epochs=10,
    batch_size=64,
    verbose=1
)

# Save the trained model
model.save('models/self_supervised_model.h5')

print('Model trained and saved to models/self_supervised_model.h5')

# Add more detailed normalization checks
print("Normalization statistics:")
print("\nNumerical features:")
print(f'Mean: {np.mean(X_num, axis=0)}')
print(f'Standard deviation: {np.std(X_num, axis=0)}')
print(f'Minimum: {np.min(X_num, axis=0)}')
print(f'Maximum: {np.max(X_num, axis=0)}')

print("\nTarget values:")
print(f'Mean: {np.mean(y_num, axis=0)}')
print(f'Standard deviation: {np.std(y_num, axis=0)}')
print(f'Minimum: {np.min(y_num, axis=0)}')
print(f'Maximum: {np.max(y_num, axis=0)}')