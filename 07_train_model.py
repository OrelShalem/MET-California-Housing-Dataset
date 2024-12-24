# 07_train_model.py

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers

def create_transformer_block(x, embed_dim, num_heads):
    """יצירת בלוק טרנספורמר"""
    # Multi-head attention
    x_reshaped = layers.Reshape((1, embed_dim))(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=embed_dim // num_heads
    )(x_reshaped, x_reshaped)
    attention_output = layers.Reshape((embed_dim,))(attention_output)
    
    # Add & Normalize
    x1 = layers.Add()([x, attention_output])
    x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
    
    # Feed-forward
    x2 = layers.Dense(embed_dim * 4, activation='relu')(x1)
    x2 = layers.Dense(embed_dim)(x2)
    
    # Add & Normalize
    out = layers.Add()([x1, x2])
    out = layers.LayerNormalization(epsilon=1e-6)(out)
    
    return out

def train_model(batch_size=256, epochs=10):
    """אימון המודל לפי גישת MET"""
    # טעינת הנתונים
    data = np.load('data/training_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    mask_train = data['mask_train']
    
    # הגדרת פרמטרים
    input_dim = X_train.shape[1]
    embed_dim = 64
    num_heads = 2
    num_transformer_blocks = 3
    
    # בניית המודל
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(embed_dim)(inputs)
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = create_transformer_block(x, embed_dim, num_heads)
    
    # Output layer
    outputs = layers.Dense(input_dim)(x)
    
    # יצירת המודל
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # הגדרת האופטימייזר
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # קומפילציה
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'],
        weighted_metrics=['mae']
    )
    
    # Custom loss mask - חישוב משקל ממוצע לכל דוגמה
    sample_weights = np.mean(mask_train, axis=1)
    
    # אימון
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        sample_weight=sample_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # שמירת המודל בפורמט החדש
    model.save('models/trained_model.keras')
    
    # שמירת היסטוריית האימון
    np.save('models/training_history.npy', history.history)
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()
    print("Model training completed and saved.")