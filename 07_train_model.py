# 07_train_model.py

import tensorflow as tf
import numpy as np

# Load the training data
data = np.load('data/training_data.npz', allow_pickle=True)
X_num = data['X_num']
X_cat = data['X_cat']
y_num = data['y_num']
y_cat = data['y_cat']

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