import tensorflow as tf
from tensorflow import keras
import numpy as np

def add_dynamic_noise(X, base_noise_level=0.1):
    """הוספת רעש דינמי לנתונים"""
    # רעש משתנה סביב רמת הבסיס
    actual_noise = base_noise_level * (1 + np.random.uniform(-0.2, 0.2))
    noise = np.random.normal(0, actual_noise, X.shape)
    return X + noise

def train_adversarial_model(epochs=10):
    """
    אימון המודל האדברסרי:
    1. טעינת הנתונים המוכנים
    2. אימון על נתונים מופרעים קבועים ודינמיים
    3. שמירת המודל המעודכן
    """
    # טעינת הנתונים המוכנים
    data = np.load('data/adversarial_data.npz', allow_pickle=True)
    X_train = data['X_train']
    X_train_noisy_sets = data['X_train_noisy_sets']
    y_train = data['y_train']
    mask_train = data['mask_train']
    
    # טעינת המודל הבסיסי
    model = keras.models.load_model('models/finetuned_model.keras')
    
    # הגדרת אופטימייזר
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    # קומפילציה
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'],
        weighted_metrics=['mae']
    )
    
    # אימון אדברסרי
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # אימון על הסטים הקבועים
        for i, X_noisy in enumerate(X_train_noisy_sets):
            print(f"\nTraining on fixed noisy set {i+1}")
            model.fit(
                X_noisy,
                y_train,
                batch_size=256,
                epochs=1,
                validation_split=0.1,
                sample_weight=np.mean(mask_train, axis=1),
                verbose=1
            )
        
        # אימון על נתונים עם רעש דינמי
        print("\nTraining on dynamic noise")
        X_dynamic_noise = add_dynamic_noise(X_train)
        model.fit(
            X_dynamic_noise,
            y_train,
            batch_size=256,
            epochs=1,
            validation_split=0.1,
            sample_weight=np.mean(mask_train, axis=1),
            verbose=1
        )
    
    # שמירת המודל
    model.save('models/adversarial_model.keras')
    print("\nAdversarial model saved successfully")
    
    return model

if __name__ == "__main__":
    model = train_adversarial_model() 