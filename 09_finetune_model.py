# 09_finetune_model.py
"""
Fine-tuning the trained model:
1. Loading the trained model
2. Loading training data
3. Continuing training with a lower learning rate
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

def reconstruction_loss(y_true, y_pred, mask):
    """MET's reconstruction loss"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_mean(mask * tf.square(y_true - y_pred))

def finetune_model():
    """
    Fine-tuning the trained model:
    1. Loading the trained model
    2. Loading training data
    3. Continuing training with a lower learning rate
    """
    # Load the trained model
    model = keras.models.load_model('models/best_model_run_1.keras')
    
    # Load training data
    data = np.load('data/training_data.npz')
    X_train = tf.cast(data['X_train'], tf.float32)
    y_train = tf.cast(data['y_train'], tf.float32)
    mask_train = tf.cast(data['mask_train'], tf.float32)
    
    # Set new optimizer with lower learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    # Custom training step
    @tf.function
    def train_step(x, y, mask):
        with tf.GradientTape() as tape:
            y_pred = model([x, mask], training=True)
            loss = reconstruction_loss(y, y_pred, mask)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # Training metrics
    train_loss_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()
    
    # Split into train/validation
    val_size = int(len(X_train) * 0.1)
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    # Convert indices to tensors
    train_idx = tf.convert_to_tensor(train_idx)
    val_idx = tf.convert_to_tensor(val_idx)
    
    # Split data into train/validation
    X_train_split = tf.gather(X_train, train_idx)
    y_train_split = tf.gather(y_train, train_idx)
    mask_train_split = tf.gather(mask_train, train_idx)
    
    X_val = tf.gather(X_train, val_idx)
    y_val = tf.gather(y_train, val_idx)
    mask_val = tf.gather(mask_train, val_idx)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(10):
        # Reset metrics
        train_loss_metric.reset_states()
        val_loss_metric.reset_states()
        
        print(f"\nEpoch {epoch+1}/10")
        
        # Progress bar
        progbar = tf.keras.utils.Progbar(
            target=1, 
            width=30,
            stateful_metrics=['loss', 'val_loss']
        )
        
        # Training
        loss = train_step(X_train_split, y_train_split, mask_train_split)
        train_loss_metric.update_state(loss)
        
        # Validation
        val_pred = model.predict([X_val, mask_val])
        val_loss = reconstruction_loss(y_val, val_pred, mask_val)
        val_loss_metric.update_state(val_loss)
        
        # Update progress bar
        values = [
            ('loss', float(train_loss_metric.result())),
            ('val_loss', float(val_loss_metric.result()))
        ]
        progbar.update(1, values=values)
        
        # Early stopping
        if val_loss_metric.result() < best_val_loss:
            best_val_loss = val_loss_metric.result()
            model.save('models/finetuned_model.keras')
            patience_counter = 0
            print("\nSaved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
    
    print("\nModel fine-tuning completed and saved to models/finetuned_model.keras")
    return model

if __name__ == "__main__":
    model = finetune_model()