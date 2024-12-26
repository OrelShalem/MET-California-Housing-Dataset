# 14_finetune_adversarial.py
"""
This script fine-tunes the adversarial model.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

def reconstruction_loss(y_true, y_pred, mask):
    """MET's reconstruction loss"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_mean(mask * tf.square(y_true - y_pred))

def mae_metric(y_true, y_pred, mask):
    """Mean Absolute Error on masked values"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_mean(mask * tf.abs(y_true - y_pred))

def accuracy_metric(y_true, y_pred, mask):
    """Calculate accuracy on masked values"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    
    correct_predictions = tf.cast(
        tf.abs(y_true - y_pred) < 0.5,
        tf.float32
    )
    return tf.reduce_sum(mask * correct_predictions) / tf.reduce_sum(mask)

def adversarial_loss(model, x, y, mask, epsilon=0.1):
    """MET's adversarial loss"""
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    mask = tf.cast(mask, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model([x, mask], training=True)
        loss = reconstruction_loss(y, pred, mask)
    
    grad = tape.gradient(loss, x)
    delta = epsilon * tf.sign(grad)
    x_adv = x + delta
    
    pred_adv = model([x_adv, mask], training=True)
    return reconstruction_loss(y, pred_adv, mask)

def finetune_adversarial_model(epochs=10):
    """Fine-tuning the adversarial model"""
    # Load data
    data = np.load('data/adversarial_data.npz', allow_pickle=True)
    X_train = tf.cast(data['X_train'], tf.float32)
    y_train = tf.cast(data['y_train'], tf.float32)
    mask_train = tf.cast(data['mask_train'], tf.float32)
    
    # Load adversarial model
    model = keras.models.load_model('models/adversarial_model.keras')
    
    # Define optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    
    # Metrics
    train_loss = tf.keras.metrics.Mean(name='total_loss')
    rec_loss_metric = tf.keras.metrics.Mean(name='rec_loss')
    adv_loss_metric = tf.keras.metrics.Mean(name='adv_loss')
    train_mae_metric = tf.keras.metrics.Mean(name='mae')
    train_acc_metric = tf.keras.metrics.Mean(name='accuracy')
    
    @tf.function
    def train_step(x, y, mask):
        with tf.GradientTape() as tape:
            y_pred = model([x, mask], training=True)
            rec_loss = reconstruction_loss(y, y_pred, mask)
            adv_loss = adversarial_loss(model, x, y, mask)
            total_loss = rec_loss + 0.1 * adv_loss
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss.update_state(total_loss)
        rec_loss_metric.update_state(rec_loss)
        adv_loss_metric.update_state(adv_loss)
        train_mae_metric.update_state(mae_metric(y, y_pred, mask))
        train_acc_metric.update_state(accuracy_metric(y, y_pred, mask))
        
        return total_loss, rec_loss, adv_loss
    
    # Fine-tuning loop
    best_loss = float('inf')
    for epoch in range(epochs):
        # Reset metrics
        train_loss.reset_states()
        rec_loss_metric.reset_states()
        adv_loss_metric.reset_states()
        train_mae_metric.reset_states()
        train_acc_metric.reset_states()
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        # Training
        total_loss, rec_loss, adv_loss = train_step(X_train, y_train, mask_train)
        
        # Calculate time
        time_per_epoch = time.time() - start_time
        time_remaining = time_per_epoch * (epochs - epoch - 1)
        
        # Print metrics
        print(f"Total Loss   : {float(train_loss.result()):.4f}")
        print(f"Rec Loss     : {float(rec_loss_metric.result()):.4f}")
        print(f"Adv Loss     : {float(adv_loss_metric.result()):.4f}")
        print(f"MAE          : {float(train_mae_metric.result()):.4f}")
        print(f"Accuracy     : {float(train_acc_metric.result()):.4f}")
        print(f"Time/epoch   : {time_per_epoch:.1f}s")
        print(f"Remaining    : {time_remaining:.1f}s")
        
        # Save the best model
        if train_loss.result() < best_loss:
            best_loss = train_loss.result()
            model.save('models/finetuned_adversarial_model.keras', save_format='tf')
            print("\nSaved new best model!")
    
    return model

if __name__ == "__main__":
    model = finetune_adversarial_model()
