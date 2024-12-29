# 12_train_adversarial_model.py
"""
Training an adversarial model:
1. Loading the model
2. Creating adversarial samples
3. Training with reconstruction and adversarial loss
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
    """Mean Absolute Error על ערכים ממוסכים"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_mean(mask * tf.abs(y_true - y_pred))

def accuracy_metric(y_true, y_pred, mask):
    """Calculate accuracy on masked values"""
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

def adversarial_loss(model, x, y, mask, epsilon=0.1):
    """
    Create adversarial samples based on MET:
    1. Calculate gradient of loss with respect to input
    2. Create perturbation in the gradient direction
    3. Calculate loss on the perturbed sample
    The process is:
    - Gradient (grad) shows the direction to change the input to increase the loss
    - tf.sign(grad) gives +1 or -1 depending on the gradient direction
    - epsilon (0.1) determines how large the perturbation will be delta is the perturbation itself
    - x_adv are the adversarial samples
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    mask = tf.cast(mask, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model([x, mask], training=True)
        loss = reconstruction_loss(y, pred, mask)
    
    # Create adversarial perturbation
    grad = tape.gradient(loss, x)
    delta = epsilon * tf.sign(grad)
    x_adv = x + delta
    
    # Calculate loss on the perturbed sample
    pred_adv = model([x_adv, mask], training=True)
    return reconstruction_loss(y, pred_adv, mask)

def train_adversarial_model(epochs=10):
    """
    Training the adversarial model:
    1. Load the base model
    2. Train with a combination of reconstruction loss and adversarial loss
    """
    # Load data
    data = np.load('data/adversarial_data.npz', allow_pickle=True)
    X_train = tf.cast(data['X_train'], tf.float32)
    y_train = tf.cast(data['y_train'], tf.float32)
    mask_train = tf.cast(data['mask_train'], tf.float32)
    
    # Load the base model
    model = keras.models.load_model('models/best_model_run_1.keras')
    
        # Set optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    # Training metrics
    train_loss = tf.keras.metrics.Mean(name='total_loss')
    rec_loss_metric = tf.keras.metrics.Mean(name='rec_loss')
    adv_loss_metric = tf.keras.metrics.Mean(name='adv_loss')
    train_mae_metric = tf.keras.metrics.Mean(name='mae')
    train_acc_metric = tf.keras.metrics.Mean(name='accuracy')
    
    # Custom training step
    @tf.function
    def train_step(x, y, mask):
        with tf.GradientTape() as tape:
            # Reconstruction loss
            y_pred = model([x, mask], training=True)
            rec_loss = reconstruction_loss(y, y_pred, mask)
            
            # Adversarial loss
            adv_loss = adversarial_loss(model, x, y, mask)
            
            # Total loss
            total_loss = rec_loss + 0.1 * adv_loss  # lambda=0.1
            
        # Update weights
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss.update_state(total_loss)
        rec_loss_metric.update_state(rec_loss)
        adv_loss_metric.update_state(adv_loss)
        train_mae_metric.update_state(mae_metric(y, y_pred, mask))
        train_acc_metric.update_state(accuracy_metric(y, y_pred, mask))
        
        return total_loss, rec_loss, adv_loss
    
    # Training loop
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
        
        # Progress bar
        progbar = tf.keras.utils.Progbar(
            target=1,
            width=30,
            stateful_metrics=['total_loss', 'rec_loss', 'adv_loss', 'mae', 'acc']
        )
        
        # Training
        total_loss, rec_loss, adv_loss = train_step(X_train, y_train, mask_train)
        
        # Calculate time
        time_per_epoch = time.time() - start_time
        time_remaining = time_per_epoch * (epochs - epoch - 1)
        
        # Update progress bar
        values = [
            ('total_loss', float(train_loss.result())),
            ('rec_loss', float(rec_loss_metric.result())),
            ('adv_loss', float(adv_loss_metric.result())),
            ('mae', float(train_mae_metric.result())),
            ('acc', float(train_acc_metric.result()))
        ]
        progbar.update(1, values=values)
        
        print(f"\nTime per epoch: {time_per_epoch:.1f}s")
        print(f"Time remaining: {time_remaining:.1f}s")
        
        # Save the best model
        if train_loss.result() < best_loss:
            best_loss = train_loss.result()
            model.save('models/adversarial_model.keras', save_format='tf')
            print("\nSaved new best model!")
    
    print("\nFinal Statistics:")
    print(f"Total Loss   : {float(train_loss.result()):.4f}")
    print(f"Rec Loss     : {float(rec_loss_metric.result()):.4f}")
    print(f"Adv Loss     : {float(adv_loss_metric.result()):.4f}")
    print(f"MAE          : {float(train_mae_metric.result()):.4f}")
    print(f"Accuracy     : {float(train_acc_metric.result()):.4f}")
    
    return model

if __name__ == "__main__":
    model = train_adversarial_model() 