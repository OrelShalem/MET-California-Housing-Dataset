# 07_train_model.py
# This script is designed to train the model.


import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from collections import defaultdict
from keras import layers
from keras.models import load_model
import json

def reconstruction_loss(y_true, y_pred, mask):
    """MET's reconstruction loss
    Calculates loss only on masked values
    Uses mean squared error (MSE)
    Ignores masked values
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_mean(mask * tf.square(y_true - y_pred))

def mae_metric(y_true, y_pred, mask):
    """Mean Absolute Error on masked values
    Calculates mean absolute error
    Only considers masked values
    Used for quality assessment
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_mean(mask * tf.abs(y_true - y_pred))

def accuracy_metric(y_true, y_pred, mask):
    """Calculate accuracy on masked values
    Calculates accuracy
    Only considers masked values
    Used for quality assessment
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    
    # Calculate accuracy on masked values
    correct_predictions = tf.cast(
        tf.abs(y_true - y_pred) < 0.5,  # threshold of 0.5
        tf.float32
    )
    accuracy = tf.reduce_sum(mask * correct_predictions) / tf.reduce_sum(mask)
    return accuracy
"""
parameters:
batch_size: number of samples per batch
epochs: number of epochs to train the model
validation_split: fraction of data to use for validation
num_runs: number of times to run the training process
"""

def train_model(batch_size=256, epochs=10, validation_split=0.2, num_runs=1):
    """Train the model multiple times and save statistics"""
    # Dictionary to store results from all runs
    all_metrics = defaultdict(list)
    
    # Load model architecture
    with open('models/model_architecture.json', 'r') as json_file:
        model_architecture = json_file.read()  # reads as string instead of converting to dict
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        
        # Load data
        data = np.load('data/training_data.npz')
        X = data['X_train']
        y = data['y_train']
        mask = data['mask_train']
        
        # Split into train/validation
        val_size = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        
        # Split data into train/validation
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        mask_train, mask_val = mask[train_idx], mask[val_idx]
        
        # Convert to tensorflow
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
        mask_train = tf.convert_to_tensor(mask_train, dtype=tf.float32)
        mask_val = tf.convert_to_tensor(mask_val, dtype=tf.float32)
        
        # Create model from saved architecture
        model = tf.keras.models.model_from_json(model_architecture)
        
        # Set optimizer and compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer)
        
        # Training metrics
        train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        train_mae_metric = tf.keras.metrics.Mean(name='train_mae')
        train_acc_metric = tf.keras.metrics.Mean(name='train_accuracy')
        val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        val_mae_metric = tf.keras.metrics.Mean(name='val_mae')
        val_acc_metric = tf.keras.metrics.Mean(name='val_accuracy')
        
        @tf.function
        def train_step(x, y, mask):
            """Train step function
            Performs one training step
            Calculates gradients and updates weights
            Updates performance metrics
            """
            with tf.GradientTape() as tape:
                y_pred = model([x, mask], training=True)
                loss = reconstruction_loss(y, y_pred, mask)
                
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update metrics
            train_loss_metric.update_state(loss)
            train_mae_metric.update_state(mae_metric(y, y_pred, mask))
            train_acc_metric.update_state(accuracy_metric(y, y_pred, mask))
            return loss
        
        @tf.function
        def val_step(x, y, mask):
            """Validation step function
            Performs validation step
            Calculates performance metrics on validation set
            """
            y_pred = model([x, mask], training=False)
            val_loss = reconstruction_loss(y, y_pred, mask)
            val_loss_metric.update_state(val_loss)
            val_mae_metric.update_state(mae_metric(y, y_pred, mask))
            val_acc_metric.update_state(accuracy_metric(y, y_pred, mask))
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # Reset metrics
            train_loss_metric.reset_states()
            train_mae_metric.reset_states()
            train_acc_metric.reset_states()
            val_loss_metric.reset_states()
            val_mae_metric.reset_states()
            val_acc_metric.reset_states()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            start_time = time.time()
            
            # Progress bar
            progbar = tf.keras.utils.Progbar(
                target=1, 
                width=30,
                stateful_metrics=['loss', 'mae', 'acc', 'val_loss', 'val_mae', 'val_acc']
            )
            
            # Training & Validation
            current_loss = train_step(X_train, y_train, mask_train)
            val_step(X_val, y_val, mask_val)

            # Calculate time per epoch
            time_per_epoch = time.time() - start_time
            time_remaining = time_per_epoch * (epochs - epoch - 1)
            
            # Update progress bar
            values = [
                ('loss', float(train_loss_metric.result())),
                ('mae', float(train_mae_metric.result())),
                ('acc', float(train_acc_metric.result())),
                ('val_loss', float(val_loss_metric.result())),
                ('val_mae', float(val_mae_metric.result())),
                ('val_acc', float(val_acc_metric.result()))
            ]
            progbar.update(1, values=values)
            
            print(f"\nTime per epoch: {time_per_epoch:.1f}s")
            print(f"Time remaining: {time_remaining:.1f}s")
            
            # Save the best model
            if val_loss_metric.result() < best_val_loss:
                best_val_loss = val_loss_metric.result()
                model.save(f'models/best_model_run_{run+1}.keras', save_format='tf')
                print("\nSaved new best model!")
        
        # Save metrics from current run
        all_metrics['train_loss'].append(float(train_loss_metric.result()))
        all_metrics['train_mae'].append(float(train_mae_metric.result()))
        all_metrics['train_acc'].append(float(train_acc_metric.result()))
        all_metrics['val_loss'].append(float(val_loss_metric.result()))
        all_metrics['val_mae'].append(float(val_mae_metric.result()))
        all_metrics['val_acc'].append(float(val_acc_metric.result()))
    
    # Print summary statistics
    print("\nFinal Statistics over", num_runs, "runs:")
    for metric_name, values in all_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric_name:10s}: {mean:.4f} ± {std:.4f}")
    
    # Add success rate summary
    print("\nSuccess Rates:")
    print(f"Training Success Rate: {float(train_acc_metric.result())*100:.2f}%")
    print(f"Validation Success Rate: {float(val_acc_metric.result())*100:.2f}%")
    
    # Add detailed performance
    print("\nDetailed Performance:")
    print(f"Training MAE: {float(train_mae_metric.result()):.4f}")
    print(f"Validation MAE: {float(val_mae_metric.result()):.4f}")
    print(f"Average prediction error: ±{float(val_mae_metric.result()):.4f} units")
    
    # Add comparison between training and validation
    train_acc = float(train_acc_metric.result())*100
    val_acc = float(val_acc_metric.result())*100
    diff = abs(train_acc - val_acc)
    print(f"\nDifference between training and validation accuracy: {diff:.2f}%")
    if diff < 5:
        print("✅ Model shows good generalization (difference < 5%)")
    else:
        print("⚠️ Model might be overfitting (difference > 5%)")

    return model, all_metrics

if __name__ == "__main__":
    model, metrics = train_model()