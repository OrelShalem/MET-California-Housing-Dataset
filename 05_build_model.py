# 05_build_model.py
# This script is designed to build the model architecture.

import tensorflow as tf
from keras import layers, models

def create_transformer_block(x, embed_dim, ff_dim, num_heads):
    """
    Create a transformer block:
    Convert input to 3D tensor
    Multi-Head Attention layer
    Feed-Forward network
    Residual Connections and Normalization
    """
    # Reshape the input to 3D tensor for attention
    x_reshaped = layers.Reshape((1, embed_dim))(x)
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=embed_dim // num_heads  # חלוקה במספר הראשים
    )(x_reshaped, x_reshaped)
    
    # Reshape back to 2D
    attention_output = layers.Reshape((embed_dim,))(attention_output)
    
    # Add & Normalize
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    ffn = layers.Dense(ff_dim, activation='relu')(x)
    ffn = layers.Dense(embed_dim)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    return x
"""
Parameters:
input_dim: Input dimension
embed_dim: Embedding dimension
ff_dim: Feed-Forward network dimension
num_heads: Number of attention heads
model_depth_enc: Number of encoder layers
model_depth_dec: Number of decoder layers
"""

def create_model(input_dim, embed_dim=64, ff_dim=64, num_heads=2, 
                model_depth_enc=6, model_depth_dec=1):
    """
    MET model architecture:
    1. Encoder-Decoder transformer
    2. Uses multiple transformer layers
    3. Input layer
    4. Initial embedding layer
    5. Encoder blocks
    6. Decoder blocks
    7. Output layer
    """
    # Input layers
    inputs = layers.Input(shape=(input_dim,))
    mask_input = layers.Input(shape=(input_dim,))
    
    # Initial embedding
    x = layers.Dense(embed_dim)(inputs)
    
    # Encoder blocks
    for _ in range(model_depth_enc):
        x = create_transformer_block(x, embed_dim, ff_dim, num_heads)
    
    # Decoder blocks
    for _ in range(model_depth_dec):
        x = create_transformer_block(x, embed_dim, ff_dim, num_heads)
    
    # Output layer
    outputs = layers.Dense(input_dim)(x)
    
    # Build model
    model = models.Model(inputs=[inputs, mask_input], outputs=outputs)
    
    return model

if __name__ == "__main__":
    # Create the model with the recommended parameters
    model = create_model(
        input_dim=8,
        embed_dim=64,
        ff_dim=64,
        num_heads=2,
        model_depth_enc=6,
        model_depth_dec=1
    )
    
    # Display the model summary
    model.summary()
    
    # Save the model architecture
    model_json = model.to_json()
    with open('models/model_architecture.json', 'w') as json_file:
        json_file.write(model_json)
    
    print('Model architecture created and saved.')