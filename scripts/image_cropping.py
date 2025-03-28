"""
This module crops images from a source folder and saves the cropped images into an output folder. 
For each image, a random top-left corner is chosen and a fixed crop size (default 600x600) is extracted.
"""

import os
from PIL import Image
import random

def crop_images(folder_path, output_folder, desired_width=600, desired_height=600, max_images=None, progress_bar=None):
    """
    Crop images from the specified folder and save the cropped version in the output folder.
    
    The function selects a random top-left corner for cropping a sub-region of the given dimensions.
    Optionally updates a global progress bar if provided.
    
    Args:
        folder_path (str): Path to the folder containing the source images.
        output_folder (str): Path to the folder where cropped images will be saved.
        desired_width (int, optional): Width of the cropped image. Defaults to 600.
        desired_height (int, optional): Height of the cropped image. Defaults to 600.
        max_images (int, optional): Maximum number of images to process. Defaults to None (process all images).
        progress_bar: Optional global progress bar (e.g., a tqdm instance) to update.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_ext = ('jpg', 'jpeg', 'png')
    all_filenames = sorted(os.listdir(folder_path))
    image_files = [fn for fn in all_filenames if fn.lower().endswith(valid_ext)]
    if max_images is not None:
        image_files = image_files[:max_images]

    for filename in image_files:
        # print(f'Processing image: {filename}')
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            max_x = img_width - desired_width
            max_y = img_height - desired_height
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
            cropped_img = img.crop((x, y, x + desired_width, y + desired_height))
            output_path = os.path.join(output_folder, filename)
            cropped_img.save(output_path)
        if progress_bar:
            progress_bar.update(1)

    print('All images have been cropped and saved to the output folder.')

if __name__ == "__main__":
    folder_path = 'data/raw_image_dataset'
    output_folder = 'data/cropped_image_dataset'
    crop_images(folder_path, output_folder)
