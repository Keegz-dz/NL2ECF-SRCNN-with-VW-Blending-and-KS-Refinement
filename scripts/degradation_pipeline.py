"""
This module provides functions to degrade high-resolution images by applying three different 
degradation techniques: Gaussian blur, Poisson noise, and isotropic blur. The functions first 
convert the input image from BGR to YCrCb color space, apply the degradation to the Y channel, and then 
convert it back to BGR format.
"""

import cv2
import numpy as np
import os

def ycrcb_format_conversion(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert an image from BGR to YCrCb color space.
    
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        
    Returns:
        np.ndarray: Image in YCrCb color space.
    """
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return img_ycrcb

def gaussian_blur(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to the Y channel of an image.
    
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        
    Returns:
        tuple: Degraded image in BGR format and a string abbreviation 'GB' for Gaussian Blur.
    """
    img_ycrcb = ycrcb_format_conversion(img_bgr)
    sigma = 2.5
    kernel_size = (5, 5)
    y_channel = img_ycrcb[:, :, 0] 
    y_channel_blur = cv2.GaussianBlur(y_channel, kernel_size, sigma)
    img_ycrcb[:, :, 0] = y_channel_blur
    gaussian_blur_img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    return gaussian_blur_img, 'GB'

def poisson_noise(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply Poisson noise to the Y channel of an image.
    
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        
    Returns:
        tuple: Noisy image in BGR format and a string abbreviation 'PN' for Poisson Noise.
    """
    img_ycrcb = ycrcb_format_conversion(img_bgr)
    y_channel = img_ycrcb[:, :, 0]
    noise_mask = np.random.poisson(y_channel)
    noisy_y_channel = np.clip(y_channel + noise_mask, 0, 255).astype(np.uint8)
    img_ycrcb[:, :, 0] = noisy_y_channel
    poisson_noise_img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    return poisson_noise_img, 'PN'

def isotropic_blur(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply isotropic blur to the Y channel of an image.
    
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        
    Returns:
        tuple: Blurred image in BGR format and a string abbreviation 'IB' for Isotropic Blur.
    """
    img_ycrcb = ycrcb_format_conversion(img_bgr)
    kernel_size = (5, 5)
    y_channel = img_ycrcb[:, :, 0]
    blurred_y_channel = cv2.blur(y_channel, kernel_size)
    img_ycrcb[:, :, 0] = blurred_y_channel
    isotropic_blur_img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    return isotropic_blur_img, 'IB'

if __name__ == '__main__':
    img_path = 'data/raw_image_dataset/0001.png'
    img_bgr = cv2.imread(img_path)
    
    output_dir = 'test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    degraded_img, operation = gaussian_blur(img_bgr)
    cv2.imwrite(os.path.join(output_dir, f'{operation}.png'), degraded_img)

    degraded_img, operation = poisson_noise(img_bgr)
    cv2.imwrite(os.path.join(output_dir, f'{operation}.png'), degraded_img)

    degraded_img, operation = isotropic_blur(img_bgr)
    cv2.imwrite(os.path.join(output_dir, f'{operation}.png'), degraded_img)
