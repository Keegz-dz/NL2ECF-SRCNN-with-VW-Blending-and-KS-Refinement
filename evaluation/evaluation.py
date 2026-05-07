"""
Evaluation script for comparing NL2ECF-SRCNN model performance against original SRCNN model.
This script evaluates PSNR and SSIM metrics across the entire dataset and provides
comparative analysis between the custom and original models.
"""

import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so we can import the 'model' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_architecture import NL2ECF_SRCNN_model, original_SRCNN_Model
from model.model_prediction import load_image, preprocess_image, model_predictions, nl2ecf_postprocess
from model.quality_metrics import evaluate_metrics


def load_original_srcnn_model(weights="logs/original_srcnn_model.h5"):
    """
    Load the original SRCNN model for comparison.

    Args:
        weights (str): Path to the trained weights file

    Returns:
        model: Loaded original SRCNN model
    """
    model = original_SRCNN_Model()

    # Try to load trained weights, fallback to untrained if not found
    try:
        model.load_weights(weights)
        print(f"Original SRCNN model loaded with trained weights: {weights}")
    except Exception as e:
        print(f"Could not load trained weights ({e})")
        print("Using untrained original SRCNN model for comparison")

    return model


def load_custom_model(model_path="logs/nl2ecf_srcnn_model.h5"):
    """
    Load the custom NL2ECF-SRCNN model.

    Args:
        model_path (str): Path to the trained model weights

    Returns:
        model: Loaded NL2ECF-SRCNN model
    """
    model = NL2ECF_SRCNN_model()
    model.load_weights(model_path)
    print(f"Custom NL2ECF-SRCNN model loaded from {model_path}")
    return model


def load_dataset_pairs(json_path="data/hr_lr_pairs.json"):
    """
    Load the dataset pairs from the JSON file.

    Args:
        json_path (str): Path to the JSON file containing HR-LR pairs

    Returns:
        dict: Dictionary mapping HR image names to list of LR image names
    """
    with open(json_path, 'r') as f:
        pairs = json.load(f)
    print(f"Loaded dataset pairs for {len(pairs)} high-resolution images")
    return pairs


def evaluate_single_image(hr_path, lr_path, custom_model, original_model=None):
    """
    Evaluate a single image pair with both models.

    Args:
        hr_path (str): Path to high-resolution ground truth image
        lr_path (str): Path to low-resolution input image
        custom_model: Loaded NL2ECF-SRCNN model
        original_model: Loaded original SRCNN model (optional)

    Returns:
        dict: Dictionary containing metrics for both models
    """
    results = {}

    # Load images
    hr_image = load_image(hr_path)
    lr_image = load_image(lr_path)

    # Evaluate custom model
    try:
        # Preprocess for custom model
        preprocessed_image, ycbcr_image = preprocess_image(lr_image)
        predicted_image = model_predictions(custom_model, preprocessed_image)
        custom_output = nl2ecf_postprocess(predicted_image, ycbcr_image)

        # Calculate metrics
        custom_metrics = evaluate_metrics(hr_image, custom_output)
        results['custom'] = custom_metrics
    except Exception as e:
        print(f"Error evaluating custom model on {lr_path}: {e}")
        results['custom'] = {'PSNR': 0, 'MSE': 0, 'SSIM': 0}

    # Evaluate original model (if provided)
    if original_model is not None:
        try:
            # For original SRCNN, we need to handle the fixed input/output sizes
            # Original SRCNN: 64x64 input -> 52x52 output (due to valid padding)

            # Convert to YCbCr and extract Y channel
            ycbcr_lr = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)
            Y_lr = ycbcr_lr[:, :, 0]

            # Resize Y channel to 64x64 for original SRCNN input
            Y_resized = cv2.resize(Y_lr, (64, 64), interpolation=cv2.INTER_CUBIC)
            Y_normalized = Y_resized.astype(float) / 255.0
            Y_normalized = np.expand_dims(Y_normalized, axis=0)
            Y_normalized = np.expand_dims(Y_normalized, axis=-1)

            # Predict with original SRCNN
            predicted_y = original_model.predict(Y_normalized, batch_size=1, verbose=0)
            predicted_y = predicted_y[0, :, :, 0] * 255  # Shape: (52, 52)
            predicted_y = np.clip(predicted_y, 0, 255)

            # Resize predicted output back to original LR dimensions for fair comparison
            predicted_y_resized = cv2.resize(predicted_y, (lr_image.shape[1], lr_image.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Reconstruct full image
            original_ycbcr = ycbcr_lr.copy()
            original_ycbcr[:, :, 0] = predicted_y_resized.astype(np.uint8)
            original_output = cv2.cvtColor(original_ycbcr, cv2.COLOR_YCrCb2BGR)

            # Calculate metrics using the original LR dimensions
            original_metrics = evaluate_metrics(hr_image, original_output)
            results['original'] = original_metrics

        except Exception as e:
            print(f"Error evaluating original model on {lr_path}: {e}")
            results['original'] = {'PSNR': 0, 'MSE': 0, 'SSIM': 0}
    else:
        results['original'] = {'PSNR': 0, 'MSE': 0, 'SSIM': 0}

    return results


def evaluate_dataset(dataset_pairs, custom_model, original_model=None, max_samples=None):
    """
    Evaluate both models across the entire dataset.

    Args:
        dataset_pairs (dict): Dictionary of HR-LR pairs
        custom_model: Loaded NL2ECF-SRCNN model
        original_model: Loaded original SRCNN model (optional)
        max_samples (int): Maximum number of samples to evaluate (optional)

    Returns:
        dict: Dictionary containing evaluation results
    """
    results = {
        'custom': {'PSNR': [], 'MSE': [], 'SSIM': []},
        'original': {'PSNR': [], 'MSE': [], 'SSIM': []},
        'sample_count': 0
    }

    hr_base_path = "data/high_resolution_images/"
    lr_base_path = "data/low_resolution_images/"

    # Process each HR image and its corresponding LR versions
    sample_count = 0

    for hr_name, lr_names in tqdm(dataset_pairs.items(), desc="Evaluating dataset"):
        if max_samples and sample_count >= max_samples:
            break

        hr_path = os.path.join(hr_base_path, f"HR_{hr_name}")

        # Check if HR image exists
        if not os.path.exists(hr_path):
            continue

        # Evaluate each LR version
        for lr_name in lr_names:
            lr_path = os.path.join(lr_base_path, lr_name)

            if not os.path.exists(lr_path):
                continue

            # Evaluate both models
            image_results = evaluate_single_image(hr_path, lr_path, custom_model, original_model)

            # Store results
            for model_name in ['custom', 'original']:
                if model_name in image_results:
                    for metric in ['PSNR', 'MSE', 'SSIM']:
                        if metric in image_results[model_name]:
                            results[model_name][metric].append(image_results[model_name][metric])

            sample_count += 1

        results['sample_count'] = sample_count

    return results


def calculate_averages(results):
    """
    Calculate average metrics from evaluation results.

    Args:
        results (dict): Results dictionary from evaluate_dataset

    Returns:
        dict: Dictionary with average metrics
    """
    averages = {}

    for model_name in ['custom', 'original']:
        averages[model_name] = {}
        for metric in ['PSNR', 'MSE', 'SSIM']:
            values = results[model_name][metric]
            if values:
                averages[model_name][metric] = np.mean(values)
                averages[model_name][f'{metric}_std'] = np.std(values)
            else:
                averages[model_name][metric] = 0
                averages[model_name][f'{metric}_std'] = 0

    return averages


def print_results(averages, sample_count):
    """
    Print evaluation results in a formatted manner.

    Args:
        averages (dict): Average metrics dictionary
        sample_count (int): Number of samples evaluated
    """
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({sample_count} samples)")
    print("="*60)

    print(f"\n{'Metric':<10} {'Custom Model':<15} {'Original Model':<15} {'Improvement':<12}")
    print("-"*62)

    for metric in ['PSNR', 'SSIM']:
        custom_val = averages['custom'][metric]
        original_val = averages['original'][metric]
        improvement = custom_val - original_val

        print(f"{metric:<10} {custom_val:<15.4f} {original_val:<15.4f} {improvement:<12.4f}")

    print(f"\n{'MSE':<10} {averages['custom']['MSE']:<15.4f} {averages['original']['MSE']:<15.4f}")

    # Summary
    print("\nSUMMARY:")
    if averages['custom']['PSNR'] > averages['original']['PSNR']:
        psnr_improvement = averages['custom']['PSNR'] - averages['original']['PSNR']
        print(f"✓ Custom model shows {psnr_improvement:.2f} dB better PSNR than original SRCNN")
    else:
        psnr_decline = averages['original']['PSNR'] - averages['custom']['PSNR']
        print(f"✗ Custom model shows {psnr_decline:.2f} dB worse PSNR than original SRCNN")

    if averages['custom']['SSIM'] > averages['original']['SSIM']:
        ssim_improvement = averages['custom']['SSIM'] - averages['original']['SSIM']
        print(f"✓ Custom model shows {ssim_improvement:.4f} better SSIM than original SRCNN")
    else:
        ssim_decline = averages['original']['SSIM'] - averages['custom']['SSIM']
        print(f"✗ Custom model shows {ssim_decline:.4f} worse SSIM than original SRCNN")


def plot_comparison(averages):
    """
    Create a comparison plot of the metrics.

    Args:
        averages (dict): Average metrics dictionary
    """
    metrics = ['PSNR', 'SSIM']
    custom_values = [averages['custom'][m] for m in metrics]
    original_values = [averages['original'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, custom_values, width, label='NL2ECF-SRCNN (Custom)', color='skyblue')
    bars2 = ax.bar(x + width/2, original_values, width, label='Original SRCNN', color='lightcoral')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Comparison plot saved as 'model_comparison.png'")


def main():
    """
    Main evaluation function.
    """
    print("Starting model evaluation...")
    print("\nNote: Make sure both models are trained before evaluation.")
    print("Train NL2ECF model: python -c 'from model.model_training import main; main(model_type=\"nl2ecf\")'")
    print("Train original model: python -c 'from model.model_training import main; main(model_type=\"original\")'")
    print("Or import and call: from model.model_training import main; main(model_type='nl2ecf')")

    # Load models
    print("\n1. Loading models...")
    try:
        custom_model = load_custom_model()
        original_model = load_original_srcnn_model()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Load dataset
    print("\n2. Loading dataset...")
    try:
        dataset_pairs = load_dataset_pairs()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Evaluate dataset
    print("\n3. Evaluating models on dataset...")
    try:
        results = evaluate_dataset(dataset_pairs, custom_model, original_model, max_samples=None)  
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # Calculate averages
    print("\n4. Calculating average metrics...")
    averages = calculate_averages(results)

    # Print results
    print_results(averages, results['sample_count'])

    # Create comparison plot
    print("\n5. Generating comparison plot...")
    try:
        plot_comparison(averages)
    except Exception as e:
        print(f"Error creating plot: {e}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
