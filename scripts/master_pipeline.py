"""
This script runs the entire image processing pipeline using a single global progress bar.It:
  1. Crops raw images.
  2. Generates HR-LR pairs.
  3. Prepares the dataset.
The global progress bar tracks each processed image across all steps.
"""

import os
from tqdm import tqdm

from image_cropping import crop_images
from dataset_creation import generate_and_save_low_resolution_for_folder
from dataset_preparation import prepare_data

def run_pipeline():
    # Define folder paths
    raw_folder = 'data/raw_image_dataset'
    cropped_folder = 'data/cropped_image_dataset'
    lr_output_dir = 'data/low_resolution_images'
    hr_output_dir = 'data/high_resolution_images'
    json_file = 'data/hr_lr_pairs.json'
    output_file_path = 'logs/processed_data.npz'
    
    # Ensure necessary directories exist
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)
        print(f"Created missing raw folder: {raw_folder}")
        print("Populate this folder with your images and re-run the pipeline.")
        return
    
    for folder in [cropped_folder, lr_output_dir, hr_output_dir, 'logs']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Determine the number of images (assuming the raw folder is the source)
    valid_ext = ('.jpg', '.jpeg', '.png')
    raw_images = [fn for fn in os.listdir(raw_folder) if fn.lower().endswith(valid_ext)]
    n_images = len(raw_images)
    
    if n_images == 0:
        print(f"No images found in {raw_folder}. Populate it and re-run the pipeline.")
        return

    # Each of the three steps processes the same number of images.
    total_steps = n_images * 3  # cropping, dataset creation, dataset preparation

    # Create a global progress bar
    with tqdm(total=total_steps, desc="Overall Pipeline Progress") as global_bar:
        print("\nStarting Image Cropping...")
        crop_images(raw_folder, cropped_folder, progress_bar=global_bar)
    
        print("\nStarting Dataset Creation...")
        generate_and_save_low_resolution_for_folder(cropped_folder, lr_output_dir, hr_output_dir, json_file, scale_factor=2, counter=None, progress_bar=global_bar)
    
        print("\nStarting Dataset Preparation...")
        prepare_data(json_file, output_file_path, limit=None, progress_bar=global_bar)

if __name__ == '__main__':
    run_pipeline()
