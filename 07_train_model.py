# 07_train_model.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import time
from collections import defaultdict

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
    """חישוב דיוק (accuracy) על ערכים ממוסכים"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    
    # נחשב את הדיוק רק על הערכים הממוסכים
    correct_predictions = tf.cast(
        tf.abs(y_true - y_pred) < 0.5,  # threshold של 0.5
        tf.float32
    )
    accuracy = tf.reduce_sum(mask * correct_predictions) / tf.reduce_sum(mask)
    return accuracy

def train_model(batch_size=256, epochs=10, validation_split=0.2, num_runs=1):
    """אימון המודל מספר פעמים ושמירת סטטיסטיקות"""
    # מילון לשמירת תוצאות מכל הריצות
    all_metrics = defaultdict(list)
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        
        # טעינת הנתונים
        data = np.load('data/training_data.npz')
        X = data['X_train']
        y = data['y_train']
        mask = data['mask_train']
        
        # חלוקה ל-train/validation
        val_size = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        
        # חלוקת הנתונים (כ-numpy arrays)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        mask_train, mask_val = mask[train_idx], mask[val_idx]
        
        # המרה ל-tensorflow
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
        mask_train = tf.convert_to_tensor(mask_train, dtype=tf.float32)
        mask_val = tf.convert_to_tensor(mask_val, dtype=tf.float32)
        
        # הגדרת פרמטרים
        input_dim = X_train.shape[1]
        embed_dim = 64
        num_heads = 2
        num_transformer_blocks = 3
        
        # בניית המודל
        inputs = layers.Input(shape=(input_dim,))
        mask_input = layers.Input(shape=(input_dim,))
        
        x = layers.Dense(embed_dim)(inputs)
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x = create_transformer_block(x, embed_dim, num_heads)
        
        # Output layer
        outputs = layers.Dense(input_dim)(x)
        
        # יצירת המודל
        model = keras.Model(inputs=[inputs, mask_input], outputs=outputs)
        
        # הגדרת האופטימייזר
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        # Training metrics
        train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        train_mae_metric = tf.keras.metrics.Mean(name='train_mae')
        train_acc_metric = tf.keras.metrics.Mean(name='train_accuracy')
        val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        val_mae_metric = tf.keras.metrics.Mean(name='val_mae')
        val_acc_metric = tf.keras.metrics.Mean(name='val_accuracy')
        
        @tf.function
        def train_step(x, y, mask):
            with tf.GradientTape() as tape:
                y_pred = model([x, mask], training=True)
                loss = reconstruction_loss(y, y_pred, mask)
                
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # עדכון מטריקות
            train_loss_metric.update_state(loss)
            train_mae_metric.update_state(mae_metric(y, y_pred, mask))
            train_acc_metric.update_state(accuracy_metric(y, y_pred, mask))
            return loss
        
        @tf.function
        def val_step(x, y, mask):
            y_pred = model([x, mask], training=False)
            val_loss = reconstruction_loss(y, y_pred, mask)
            val_loss_metric.update_state(val_loss)
            val_mae_metric.update_state(mae_metric(y, y_pred, mask))
            val_acc_metric.update_state(accuracy_metric(y, y_pred, mask))
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # איפוס מטריקות
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
            
            # חישוב זמנים
            time_per_epoch = time.time() - start_time
            time_remaining = time_per_epoch * (epochs - epoch - 1)
            
            # עדכון progress bar
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
            
            # שמירת המודל הטוב ביותר
            if val_loss_metric.result() < best_val_loss:
                best_val_loss = val_loss_metric.result()
                model.save(f'models/best_model_run_{run+1}.keras', save_format='tf')
                print("\nSaved new best model!")
        
        # שמירת המטריקות מהריצה הנוכחית
        all_metrics['train_loss'].append(float(train_loss_metric.result()))
        all_metrics['train_mae'].append(float(train_mae_metric.result()))
        all_metrics['train_acc'].append(float(train_acc_metric.result()))
        all_metrics['val_loss'].append(float(val_loss_metric.result()))
        all_metrics['val_mae'].append(float(val_mae_metric.result()))
        all_metrics['val_acc'].append(float(val_acc_metric.result()))
    
    # הדפסת סטטיסטיקות מסכמות
    print("\nFinal Statistics over", num_runs, "runs:")
    for metric_name, values in all_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric_name:10s}: {mean:.4f} ± {std:.4f}")
    
    return model, all_metrics

if __name__ == "__main__":
    model, metrics = train_model()