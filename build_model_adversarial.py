import tensorflow as tf
from tensorflow import keras
from keras import layers

class AdversarialLoss(keras.losses.Loss):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def call(self, y_true, y_pred):
        noise = tf.random.normal(shape=tf.shape(y_pred), mean=0.0, stddev=self.epsilon)
        perturbed_pred = y_pred + noise
        original_loss = tf.keras.losses.MSE(y_true, y_pred)
        adversarial_loss = tf.keras.losses.MSE(y_true, perturbed_pred)
        return original_loss + 0.5 * adversarial_loss

def build_adversarial_model(input_dim, num_categories):
    # אותה ארכיטקטורה כמו המודל המקורי
    numerical_inputs = keras.Input(shape=(input_dim,), name='numerical_inputs')
    categorical_inputs = keras.Input(shape=(1,), name='categorical_inputs')
    embedding = layers.Embedding(num_categories + 1, 2, name='category_embedding')(categorical_inputs)
    x_cat = layers.Flatten()(embedding)
    
    x = layers.Concatenate()([numerical_inputs, x_cat])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    
    numerical_outputs = layers.Dense(input_dim, name='numerical_outputs')(x)
    categorical_outputs = layers.Dense(
        num_categories + 1,
        activation='softmax',
        name='categorical_outputs'
    )(x)
    
    model = keras.Model(
        inputs=[numerical_inputs, categorical_inputs],
        outputs=[numerical_outputs, categorical_outputs]
    )
    
    # Compile with the adversarial loss
    model.compile(
        optimizer='adam',
        loss={
            'numerical_outputs': AdversarialLoss(epsilon=0.1),
            'categorical_outputs': 'sparse_categorical_crossentropy'
        }
    )
    
    return model

if __name__ == "__main__":
    # Create the adversarial model
    model = build_adversarial_model(input_dim=7, num_categories=3)
    model.summary()
    
    # Save the model with a different name from the original model
    model.save('models/self_supervised_model_adversarial.h5')
    print('Adversarial model architecture created and saved.') 