"""
This script trains the NL2ECF-SRCNN model using preprocessed data and saves the trained model.
It utilizes a TQDM progress bar to display batch-level progress during training.
After training is complete, it plots the training and validation loss curves.

Note:
    - This file is in the root directory.
    - The model architecture is imported from model/model_architecture.py.
"""

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_architecture import NL2ECF_SRCNN_model, original_SRCNN_Model
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


def train_model(data_file_path, model_file_path, model_type='nl2ecf', batch_size=16, epochs=5, validation_split=0.2):
    """
    Trains either the NL2ECF-SRCNN model or the original SRCNN model using the provided dataset.
    It uses a TQDM progress bar to provide batch-level progress feedback.

    Args:
        data_file_path (str): Path to the .npz file with training data.
        model_file_path (str): Path where the trained model will be saved.
        model_type (str): Type of model to train ('nl2ecf' for NL2ECF-SRCNN or 'original' for original SRCNN).
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        validation_split (float): Fraction of data to use for validation.

    Returns:
        history: Training history object.
    """
    input_images, target_images = load_data(data_file_path)
    print("Input images shape:", input_images.shape)
    print("Target images shape:", target_images.shape)

    if model_type.lower() == 'original':
        print("Training original SRCNN model...")
        model = original_SRCNN_Model()

        # For original SRCNN, we need to resize inputs to fixed 64x64 size
        # Note: Original SRCNN with valid padding produces 52x52 output from 64x64 input
        print("Resizing input images to 64x64 and target images to 52x52 for original SRCNN...")
        resized_inputs = []
        resized_targets = []

        for i in range(len(input_images)):
            # Resize input to 64x64
            input_resized = cv2.resize(input_images[i], (64, 64), interpolation=cv2.INTER_CUBIC)
            # Resize target to 52x52 to match SRCNN output (due to valid padding)
            target_resized = cv2.resize(target_images[i], (52, 52), interpolation=cv2.INTER_CUBIC)

            resized_inputs.append(input_resized)
            resized_targets.append(target_resized)

        input_images = np.array(resized_inputs)
        target_images = np.array(resized_targets)

        print("Resized input images shape:", input_images.shape)
        print("Resized target images shape:", target_images.shape)

    elif model_type.lower() == 'nl2ecf':
        print("Training NL2ECF-SRCNN model...")
        model = NL2ECF_SRCNN_model()
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'nl2ecf' or 'original'")

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

def main(model_type='nl2ecf', data_path=None, model_path=None, batch_size=16, epochs=10, validation_split=0.2):
    """
    Main function to train SRCNN models with specified parameters.

    Args:
        model_type (str): Type of model to train ('nl2ecf' or 'original')
        data_path (str): Path to the processed data file (default: logs/processed_data.npz)
        model_path (str): Path to save the trained model (auto-generated if None)
        batch_size (int): Batch size for training (default: 16)
        epochs (int): Number of training epochs (default: 10)
        validation_split (float): Fraction of data for validation (default: 0.2)

    Returns:
        history: Training history object
    """
    # Set default data path
    if data_path is None:
        data_path = os.path.join(os.getcwd(), 'logs', 'processed_data.npz')

    # Set default model path based on model type
    if model_path is None:
        if model_type == 'nl2ecf':
            model_path = os.path.join(os.getcwd(), 'logs', 'nl2ecf_srcnn_model_v1.h5')
        else:  # original
            model_path = os.path.join(os.getcwd(), 'logs', 'original_srcnn_model.h5')

    print(f"Training {model_type.upper()} model...")
    print(f"Data path: {data_path}")
    print(f"Model save path: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Validation split: {validation_split}")

    # Train the model with progress bar feedback.
    history = train_model(
        data_path,
        model_path,
        model_type,
        batch_size,
        epochs,
        validation_split
    )

    # Plot training and validation loss curves after training.
    plot_training_history(history)

    return history


if __name__ == '__main__':
    # Default training configuration
    main(model_type='original', epochs=30)
