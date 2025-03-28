"""
This script trains the NL2ECF-SRCNN model using preprocessed data and saves the trained model.
It utilizes a TQDM progress bar to display batch-level progress during training.
After training is complete, it plots the training and validation loss curves.

Note:
    - This file is in the root directory.
    - The model architecture is imported from model/model_architecture.py.
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from model.model_architecture import NL2ECF_SRCNN_model
from tqdm.keras import TqdmCallback


def load_data(file_path):
    """
    Loads the processed data from an .npz file and pairs the HR and LR images.
    
    Args:
        file_path (str): Path to the .npz file containing 'input_images' and 'target_images'.
    
    Returns:
        tuple: Two numpy arrays containing low-resolution images and high-resolution images.
    """
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path)
    input_images = data['input_images']
    target_images = data['target_images']
    print("Data loaded.")
    
    # Pair each HR image with its corresponding LR images.
    paired_data = []
    for i in range(target_images.shape[0]):
        hr_image = target_images[i]
        for j in range(input_images.shape[-1]):
            lr_image = input_images[i, :, :, j]
            paired_data.append((lr_image, hr_image))
    
    lr_images, hr_images = zip(*paired_data)
    lr_images = np.array(lr_images)
    hr_images = np.array(hr_images)
    
    return lr_images, hr_images


def train_model(data_file_path, model_file_path, batch_size=16, epochs=5, validation_split=0.2):
    """
    Trains the NL2ECF-SRCNN model using the provided dataset and saves the trained model.
    It uses a TQDM progress bar to provide batch-level progress feedback.
    
    Args:
        data_file_path (str): Path to the .npz file with training data.
        model_file_path (str): Path where the trained model will be saved.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        validation_split (float): Fraction of data to use for validation.
    
    Returns:
        history: Training history object.
    """
    input_images, target_images = load_data(data_file_path)
    print("Input images shape:", input_images.shape)
    print("Target images shape:", target_images.shape)
    
    model = NL2ECF_SRCNN_model()

    print("Starting training...")
    history = model.fit(
        input_images,
        target_images,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[TqdmCallback(verbose=1)]
    )

    model.save(model_file_path)
    print(f"Model training complete and saved as {model_file_path}.")
    
    return history


def plot_training_history(history):
    """
    Plots the training and validation loss curves.
    
    Args:
        history: Training history object from model.fit.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    base_directory = os.getcwd()  
    data_file_path = os.path.join(base_directory, 'logs', 'processed_data.npz')
    model_file_path = os.path.join(base_directory, 'logs', 'nl2ecf_srcnn_model.h5')
    
    # Define training parameters.
    batch_size = 16
    epochs = 10
    validation_split = 0.2

    # Train the model with progress bar feedback.
    history = train_model(data_file_path, model_file_path, batch_size, epochs, validation_split)
    
    # Plot training and validation loss curves after training.
    plot_training_history(history)
