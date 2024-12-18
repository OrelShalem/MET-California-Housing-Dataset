# 10_visualize_embeddings.py

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle

# Load the trained model
model = load_model('models/self_supervised_model.h5')

# Load the label encoder to get category names
with open('models/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
category_names = label_encoder.classes_

# Extract the embedding weights
embedding_layer = model.get_layer('embedding')
embeddings = embedding_layer.get_weights()[0]  # Shape: (num_categories + 1, embedding_dim)

# Exclude the mask token embedding (index 0) if mask_zero=True was used
embeddings = embeddings[1:]  # Adjust according to your embedding layer configuration

# Optionally normalize embeddings
# embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Visualize embeddings using PCA or t-SNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Use PCA for dimensionality reduction
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Alternatively, use t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings)

# Plot the embeddings
plt.figure(figsize=(8, 6))
for i, label in enumerate(category_names):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, label, fontsize=12)
plt.title('Embeddings Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()