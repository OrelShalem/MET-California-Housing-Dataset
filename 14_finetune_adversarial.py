import tensorflow as tf
from tensorflow import keras
import numpy as np

def finetune_adversarial_model():
    """
    Fine-tuning של המודל האדברסרי:
    1. טעינת המודל האדברסרי
    2. המשך אימון עם learning rate נמוך יותר
    3. שמירת המודל המעודכן
    """
    # טעינת הנתונים
    data = np.load('data/adversarial_data.npz', allow_pickle=True)
    X_train = data['X_train']
    X_train_noisy_sets = data['X_train_noisy_sets']
    y_train = data['y_train']
    mask_train = data['mask_train']
    
    # טעינת המודל האדברסרי
    model = keras.models.load_model('models/adversarial_model.keras')
    
    # הגדרת אופטימייזר
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'],
        weighted_metrics=['mae']
    )
    
    # אימון על כל סט נתונים מופרע בנפרד
    for epoch in range(10):
        print(f"\nEpoch {epoch+1}/10")
        
        # אימון על הנתונים המקוריים
        history_clean = model.fit(
            X_train,
            y_train,
            batch_size=256,
            epochs=1,
            validation_split=0.1,
            sample_weight=np.mean(mask_train, axis=1),
            verbose=1
        )
        
        # אימון על כל אחד מהסטים המופרעים
        for i, X_noisy in enumerate(X_train_noisy_sets):
            print(f"\nTraining on noisy set {i+1}")
            history_noisy = model.fit(
                X_noisy,  # כעת זה בצורה הנכונה
                y_train,
                batch_size=256,
                epochs=1,
                validation_split=0.1,
                sample_weight=np.mean(mask_train, axis=1),
                verbose=1
            )
    
    # שמירת המודל המעודכן
    model.save('models/finetuned_adversarial_model.keras')
    
    return model, history_clean

if __name__ == "__main__":
    model, history = finetune_adversarial_model()
