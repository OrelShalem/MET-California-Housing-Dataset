import numpy as np
from tensorflow import keras
from build_model_adversarial import AdversarialLoss

# טעינת הנתונים
data = np.load('data/training_data.npz')
X_num, X_cat = data['X_num'], data['X_cat']
y_num, y_cat = data['y_num'], data['y_cat']

# טעינת המודל האדברסרי
model = keras.models.load_model('models/self_supervised_model_adversarial_trained.h5',
                              custom_objects={'AdversarialLoss': AdversarialLoss})

# הגדרת שכבות חדשות לfine-tuning עם שמות ייחודיים
x = model.layers[-3].output
x = keras.layers.Dense(48, activation='relu', name='dense_finetune_1')(x)
x = keras.layers.Dense(24, activation='relu', name='dense_finetune_2')(x)
numerical_outputs = keras.layers.Dense(7, name='numerical_outputs_new')(x)
categorical_outputs = keras.layers.Dense(4, name='categorical_outputs_new')(x)

# יצירת מודל חדש
new_model = keras.Model(
    inputs=model.inputs,
    outputs=[numerical_outputs, categorical_outputs]
)

# קומפילציה
new_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss={
        'numerical_outputs_new': AdversarialLoss(epsilon=0.05),
        'categorical_outputs_new': 'sparse_categorical_crossentropy'
    }
)

# אימון המודל המעודכן
history = new_model.fit(
    [X_num, X_cat],
    [y_num, y_cat],
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

# הערכת המודל
test_loss = new_model.evaluate([X_num[-1000:], X_cat[-1000:]], 
                             [y_num[-1000:], y_cat[-1000:]])
print(f"Test Mean Squared Error: {test_loss[0]}")

# שמירת המודל המעודכן
new_model.save('models/self_supervised_model_adversarial_finetuned.h5') 