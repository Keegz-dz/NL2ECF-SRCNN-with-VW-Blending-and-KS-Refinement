"""
This module processes high-resolution (HR) images stored in a folder, applies different degradation 
functions to generate low-resolution (LR) images, and creates HR-LR mapping pairs in JSON format.

The degradation functions include Gaussian blur, Poisson noise, and isotropic blur, imported from the 
degradation_pipeline module. Additionally, a quality distortion is introduced by downscaling and then 
upscaling the degraded image.
"""

import cv2
import os
import json
import numpy as np

from degradation_pipeline import gaussian_blur, poisson_noise, isotropic_blur

def quality_distortion(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Apply quality distortion by resizing the image down and then back up.
    
    Args:
        image (np.ndarray): The input image in BGR format.
        scale_factor (int): Factor by which the image is downscaled and upscaled.
        
    Returns:
        np.ndarray: The distorted image with quality loss.
    """
    h, w, _ = image.shape
    new_height = h // scale_factor
    new_width = w // scale_factor
        
    down_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    up_img = cv2.resize(down_img, (w, h), interpolation=cv2.INTER_LINEAR)
    return up_img

def generate_and_save_low_resolution_for_folder(hr_images_folder: str, lr_output_dir: str, hr_output_dir: str, json_file: str, scale_factor: int = 2, counter: int = None, progress_bar=None):
    """
    Generate low resolution images from HR images using various degradation methods and save them along 
    with the HR images. A JSON mapping between HR and corresponding LR images is also saved.
    
    Args:
        hr_images_folder (str): Directory path containing HR images.
        lr_output_dir (str): Output directory for LR images.
        hr_output_dir (str): Output directory for HR images.
        json_file (str): File path to save the HR-LR mapping in JSON format.
        scale_factor (int, optional): Factor used for quality distortion. Defaults to 2.
        counter (int, optional): Maximum number of images to process. Defaults to None (process all).
        progress_bar: Optional global progress bar (e.g., a tqdm instance) to update.
    """
    if not os.path.exists(hr_output_dir):
        os.makedirs(hr_output_dir)
    if not os.path.exists(lr_output_dir):
        os.makedirs(lr_output_dir)
          
    hr_lr_pairs = {}
    valid_ext = ('.png', '.jpg', '.jpeg')
    all_filenames = sorted(os.listdir(hr_images_folder))
    image_files = [fn for fn in all_filenames if fn.lower().endswith(valid_ext)]
    if counter is not None:
        image_files = image_files[:counter]
      
    for filename in image_files:
        hr_image_path = os.path.join(hr_images_folder, filename)
        img_bgr = cv2.imread(hr_image_path)
                  
        hr_output_path = os.path.join(hr_output_dir, 'HR_' + filename)
        cv2.imwrite(hr_output_path, img_bgr)

        lr_images_list = []
        for abbreviation in ['GB', 'PN', 'IB']:
            if abbreviation == 'GB':
                degraded_img, _ = gaussian_blur(img_bgr)
            elif abbreviation == 'PN':
                degraded_img, _ = poisson_noise(img_bgr)
            elif abbreviation == 'IB':
                degraded_img, _ = isotropic_blur(img_bgr)
                      
            degraded_img_resized = quality_distortion(degraded_img, scale_factor)
            lr_filename = f'LR_{abbreviation}_' + filename
            lr_output_path = os.path.join(lr_output_dir, lr_filename)
            cv2.imwrite(lr_output_path, degraded_img_resized)
            lr_images_list.append(lr_filename)

        hr_lr_pairs[filename] = lr_images_list
        
        if progress_bar:
            progress_bar.update(1)

    with open(json_file, 'w') as jsonfile:
        json.dump(hr_lr_pairs, jsonfile, indent=4)
          
    print(f"HR-LR pairs saved to: {json_file}")

if __name__ == '__main__':
    hr_images_folder = 'data/cropped_image_dataset'  
    lr_output_dir = 'data/low_resolution_images'  
    hr_output_dir = 'data/high_resolution_images'  
    json_file = 'data/hr_lr_pairs.json'  

    counter = None  # Set to None to process all images
    generate_and_save_low_resolution_for_folder(hr_images_folder, lr_output_dir, hr_output_dir, json_file, scale_factor=2, counter=counter)
