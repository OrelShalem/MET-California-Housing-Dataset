# 09_finetune_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras import layers, models
import keras

# Load the original data
df = pd.read_csv('data/encoded_data.csv')  # This contains the unmasked data with encoded categorical features

# Prepare inputs
numerical_features = [
    'MedInc',
    'AveRooms',
    'AveBedrms',
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude'
]
categorical_features = ['AgeCategory']

X_num = df[numerical_features].values
X_cat = df[categorical_features].values
y = df['MedHouseVal'].values  # Target variable

# Split the data into training and testing sets
X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# Load the pretrained self-supervised model
model = load_model('models/self_supervised_model.h5')

# Remove the output layers of the self-supervised model
# Retain the shared hidden layers to use as a feature extractor
# Identify the layer before the outputs (e.g., 'dense_1')

# Optionally, print the model summary to identify layer names
# model.summary()

# Create a new model with the shared layers
# Assuming 'dense_1' is the last shared layer before outputs
shared_layer_output = model.get_layer('dense_1').output

# Add a new output layer for regression
price_output = layers.Dense(1, name='price_output')(shared_layer_output)

# Define the new model
model_finetuned = models.Model(inputs=model.inputs, outputs=price_output)

# Optionally, freeze the layers except the new output layer
for layer in model_finetuned.layers[:-1]:
    layer.trainable = False

# Compile the model
model_finetuned.compile(optimizer='adam', loss='mse')

# Train the model
history = model_finetuned.fit(
    {'numerical_inputs': X_num_train, 'categorical_inputs': X_cat_train},
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Evaluate the model on the test set
y_pred = model_finetuned.predict({'numerical_inputs': X_num_test, 'categorical_inputs': X_cat_test})
test_mse = mean_squared_error(y_test, y_pred)
print(f'Test Mean Squared Error: {test_mse}')

# Save the fine-tuned model
model_finetuned.save('models/fine_tuned_model.h5')

print('Fine-tuning completed and model saved to models/fine_tuned_model.h5')