"""
This module provides functions to compute quality metrics between images produced by the NL2ECF-SRCNN model.
It includes functions to compute PSNR (Peak Signal-to-Noise Ratio), MSE (Mean Squared Error), and SSIM 
(Structural Similarity Index Measure). The evaluate_metrics function converts images to grayscale (if necessary)
before computing the metrics.
"""

import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import math

def psnr(target: np.ndarray, ref: np.ndarray) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        target (np.ndarray): Target image.
        ref (np.ndarray): Reference image.
    
    Returns:
        float: PSNR value.
    """
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255. / rmse)

def mse(target: np.ndarray, ref: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between two images.
    
    Args:
        target (np.ndarray): Target image.
        ref (np.ndarray): Reference image.
    
    Returns:
        float: MSE value.
    """
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])
    return err

def compare_images(target: np.ndarray, ref: np.ndarray) -> dict:
    """
    Computes PSNR, MSE, and SSIM between two images.
    
    Args:
        target (np.ndarray): Target image.
        ref (np.ndarray): Reference image.
    
    Returns:
        dict: Dictionary with keys 'PSNR', 'MSE', and 'SSIM'.
    """
    scores = {}
    scores['PSNR'] = psnr(target, ref)
    scores['MSE'] = mse(target, ref)
    scores['SSIM'] = ssim(target, ref, multichannel=True)
    return scores

def evaluate_metrics(original_image: np.ndarray, reconstructed_image: np.ndarray) -> dict:
    """
    Evaluates quality metrics between the original and reconstructed images. If the images have 3 channels,
    they are converted to grayscale.
    
    Args:
        original_image (np.ndarray): Original high-resolution image.
        reconstructed_image (np.ndarray): Reconstructed or super-resolved image.
    
    Returns:
        dict: Dictionary containing PSNR, MSE, and SSIM scores.
    """
    if original_image.shape[-1] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if reconstructed_image.shape[-1] == 3:
        reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2GRAY)
    
    scores = compare_images(original_image, reconstructed_image)
    return scores

if __name__ == "__main__":
    original_image = cv2.imread('data/high_resolution_images/HR_0001.png')
    reconstructed_image = cv2.imread('data/low_resolution_images/LR_PN_0001.png')
    scores = evaluate_metrics(original_image, reconstructed_image)
    for metric, value in scores.items():
        print(f"{metric}: {value}")
