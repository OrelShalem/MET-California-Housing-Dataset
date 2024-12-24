# 09_finetune_model.py

import tensorflow as tf
from tensorflow import keras
import numpy as np

def finetune_model():
    """
    Fine-tuning של המודל המאומן:
    1. טעינת המודל המאומן
    2. טעינת נתוני האימון
    3. המשך אימון עם learning rate נמוך יותר
    """
    # טעינת המודל המאומן
    model = keras.models.load_model('models/trained_model.keras')
    
    # טעינת הנתונים
    data = np.load('data/training_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    mask_train = data['mask_train']
    
    # הגדרת אופטימייזר חדש עם learning rate נמוך יותר
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    # קומפילציה מחדש של המודל
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'],
        weighted_metrics=['mae']
    )
    
    # המשך אימון
    history = model.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=50,
        validation_split=0.1,
        sample_weight=np.mean(mask_train, axis=1),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # שמירת המודל המעודכן
    model.save('models/finetuned_model.keras')
    
    # שמירת היסטוריית האימון
    np.save('models/finetuning_history.npy', history.history)
    
    return model, history

if __name__ == "__main__":
    model, history = finetune_model()
    print("\nModel fine-tuning completed and saved to models/finetuned_model.keras")