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

def adversarial_loss(model, x, y, mask, epsilon=0.1):
    """
    יצירת דוגמאות אדברסריות לפי MET:
    1. חישוב הגרדיאנט של הloss ביחס לקלט
    2. יצירת הפרעה בכיוון הגרדיאנט
    3. חישוב loss על הדוגמא המופרעת
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    mask = tf.cast(mask, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model([x, mask], training=True)
        loss = reconstruction_loss(y, pred, mask)
    
    # יצירת הפרעה אדברסרית
    grad = tape.gradient(loss, x)
    delta = epsilon * tf.sign(grad)
    x_adv = x + delta
    
    # חישוב loss על הדוגמא המופרעת
    pred_adv = model([x_adv, mask], training=True)
    return reconstruction_loss(y, pred_adv, mask)

def train_adversarial_model(epochs=10):
    """
    אימון המודל האדברסרי:
    1. טעינת המודל הבסיסי
    2. אימון עם שילוב של reconstruction loss ו-adversarial loss
    """
    # טעינת הנתונים
    data = np.load('data/adversarial_data.npz', allow_pickle=True)
    X_train = tf.cast(data['X_train'], tf.float32)
    y_train = tf.cast(data['y_train'], tf.float32)
    mask_train = tf.cast(data['mask_train'], tf.float32)
    
    # טעינת המודל הבסיסי
    model = keras.models.load_model('models/best_model_run_1.keras')
    
    # הגדרת אופטימייזר
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
            
        # עדכון המשקלים
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # עדכון מטריקות
        train_loss.update_state(total_loss)
        rec_loss_metric.update_state(rec_loss)
        adv_loss_metric.update_state(adv_loss)
        train_mae_metric.update_state(mae_metric(y, y_pred, mask))
        train_acc_metric.update_state(accuracy_metric(y, y_pred, mask))
        
        return total_loss, rec_loss, adv_loss
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        # איפוס מטריקות
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
        
        # חישוב זמנים
        time_per_epoch = time.time() - start_time
        time_remaining = time_per_epoch * (epochs - epoch - 1)
        
        # עדכון progress bar
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
        
        # שמירת המודל הטוב ביותר
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