# Self-Supervised Learning for Housing Data

This repository implements a self-supervised learning approach for the California Housing dataset. The project demonstrates how to train a model to predict missing values in data using a masked value prediction technique, similar to BERT's masked language modeling but for tabular data.

## Project Overview

The project uses a neural network to learn meaningful representations of housing data by training it to reconstruct masked (missing) features. This approach is particularly valuable when dealing with incomplete data or when you want to leverage unlabeled data for pre-training before fine-tuning on a specific task.

### Features

- Self-supervised learning on tabular data
- Handling of both numerical and categorical features
- Pre-training with masked value prediction
- Fine-tuning for a specific downstream task (housing price prediction)
- Adversarial training option for improved robustness

## Data Pipeline

The project follows a step-by-step data processing and modeling pipeline:

1. **Data Loading**: Load California Housing dataset from scikit-learn
2. **Data Preparation**: Preprocess data and create categorical features
3. **Categorical Encoding**: Encode categorical features into numerical representations
4. **Masking**: Apply masking to features (hiding ~70% of values) to create training data
5. **Model Building**: Create a neural network architecture for self-supervised learning
6. **Training Data Preparation**: Prepare masked and original data for training
7. **Model Training**: Train the model to predict the original values from masked input
8. **Model Evaluation**: Evaluate the model's performance on reconstruction tasks
9. **Fine-tuning**: Adapt the pre-trained model for housing price prediction
10. **Embedding Visualization**: Visualize learned categorical embeddings

## File Structure

- `01_load_data.py`: Loads the California Housing dataset
- `02_data_preparation.py`: Preprocesses data and creates categorical features
- `03_encode_categorical.py`: Encodes categorical features using label encoding
- `04_apply_masking.py`: Applies masking to create self-supervised training data
- `05_build_model.py`: Defines the neural network architecture
- `06_prepare_training_data.py`: Prepares the input and target data for training
- `07_train_model.py`: Trains the self-supervised model
- `08_evaluate_model.py`: Evaluates the model's performance
- `09_finetune_model.py`: Fine-tunes the model for housing price prediction
- `10_visualize_embeddings.py`: Visualizes the learned embeddings

Additional advanced implementations:
- `build_model_adversarial.py`: Implements an adversarial training approach
- `07_train_model_adversarial.py`: Trains the model with adversarial loss
- `08_evaluate_model_adversarial.py`: Evaluates the adversarial model
- `09_finetune_model_adversarial.py`: Fine-tunes the adversarial model

## Model Architecture

The model consists of:

- Separate input branches for numerical and categorical features
- Embedding layers for categorical features
- Shared hidden layers for feature extraction
- Separate output heads for numerical and categorical reconstruction

```
Model Architecture:
- Numerical Input → Dense Layers
                      ↓
- Categorical Input → Embedding → Flatten → Dense Layers
                                    ↑
                                    ↓
                                Concatenate
                                    ↓
                                Dense Layers
                                    ↓
                    ┌─────────────┴────────────────┐
                    ↓                              ↓
Numerical Output (Regression)    Categorical Output (Classification)
```

## Adversarial Training

The project includes an alternative implementation with adversarial training to improve model robustness. Adversarial training works by adding small perturbations to the model's predictions during training, forcing it to learn representations that are resilient to noise.

The `AdversarialLoss` class in `build_model_adversarial.py` implements this approach, combining standard loss with the loss computed on perturbed predictions.

## Requirements

- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

## Directory Structure

The project expects the following directory structure:

```
.
├── data/               # Data directory
│   ├── raw_data.csv
│   ├── prepared_data.csv
│   ├── encoded_data.csv
│   ├── masked_data.csv
│   ├── mask_indices.csv
│   └── training_data.npz
├── models/             # Model directory
│   ├── scaler_features.pkl
│   ├── scaler_target.pkl
│   ├── label_encoder.pkl
│   ├── model_architecture.json
│   ├── self_supervised_model.h5
│   ├── self_supervised_model_adversarial.h5
│   └── fine_tuned_model.h5
├── .gitignore
├── *.py                # Python scripts
└── README.md
```

## Getting Started

1. Clone the repository
2. Create necessary directories: `mkdir -p data models`
3. Run the scripts in order (01 through 10)

Example:
```bash
python 01_load_data.py
python 02_data_preparation.py
# ... and so on
```

## The Self-Supervised Learning Approach

This project demonstrates how to use self-supervised learning for tabular data. The main idea is to:

1. Mask a significant portion (~70%) of the data
2. Train a neural network to predict the original values from the masked input
3. Use the trained model's internal representations for downstream tasks

This approach is analogous to how BERT and other transformer models are pre-trained for NLP tasks, but adapted for structured tabular data.

## Fine-tuning for Housing Price Prediction

After pre-training the model to reconstruct masked values, the learned representations can be leveraged for a specific task. The project demonstrates this by fine-tuning the model to predict median house values.

The fine-tuning process:
1. Loads the pre-trained self-supervised model
2. Removes the original output layers
3. Adds a new output layer for housing price prediction
4. Trains the model on the complete, unmasked dataset with house prices as targets

## Visualization

The `10_visualize_embeddings.py` script demonstrates how to extract and visualize the learned categorical embeddings. This can provide insights into how the model has learned to represent the categorical features in a continuous space.

## Acknowledgments

This project uses the California Housing dataset provided by scikit-learn, which is derived from the 1990 California census data.
