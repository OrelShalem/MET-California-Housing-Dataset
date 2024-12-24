# 05_build_model.py

# explain: tensorflow is used to build the model and keras is used to build the layers
import tensorflow as tf
from keras import layers, models

def create_model(num_numerical_features, num_categorical_features, num_categories):
    # Input layers
    numerical_inputs = layers.Input(shape=(num_numerical_features,), name='numerical_inputs')
    categorical_inputs = layers.Input(shape=(num_categorical_features,), name='categorical_inputs')

    # Embedding layers for categorical features 
    # explain: embedding is used to embed the categorical features into a dense vector
    # explain: input_dim is the number of categories + 1 to include the mask token at index 0
    # explain: output_dim is the dimension of the embedding
    # explain: mask_zero is used to mask the zero index
    embedding_dim = 2  # Adjust as needed
    embedding = layers.Embedding(
        input_dim=num_categories + 1,  # +1 to include the mask token at index 0
        output_dim=embedding_dim,
        mask_zero=True
    )(categorical_inputs)

    # Flatten embedding output 
    # explain: flatten is used to flatten the embedding output to a 1D vector
    embedding_flat = layers.Flatten()(embedding)

    # Concatenate numerical and categorical features
    concat = layers.Concatenate()([numerical_inputs, embedding_flat])

    # Shared hidden layers
    hidden = layers.Dense(128, activation='relu')(concat)
    hidden = layers.Dense(64, activation='relu')(hidden)
    hidden = layers.Dense(32, activation='relu')(hidden)

    # Output layers for reconstructing numerical features
    numerical_outputs = layers.Dense(num_numerical_features, name='numerical_outputs')(hidden)

    # Output layers for reconstructing categorical features
    categorical_outputs = layers.Dense(
        num_categories + 1,  # +1 to include the mask token
        activation='softmax',
        name='categorical_outputs'
    )(hidden)

    # Define the model
    model = models.Model(
        inputs=[numerical_inputs, categorical_inputs],
        outputs=[numerical_outputs, categorical_outputs]
    )

    return model

# Calculate num_categories based on label encoding
num_categories = 3  # Adjust based on the data; in this case, we have 'New', 'Mid', 'Old', so 3 categories

# Save the model architecture (weights will be trained later)
model = create_model(num_numerical_features=7, num_categorical_features=1, num_categories=num_categories)

# Save model architecture to JSON (optional)
model_json = model.to_json()
with open('models/model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

print('Model architecture created and saved.')