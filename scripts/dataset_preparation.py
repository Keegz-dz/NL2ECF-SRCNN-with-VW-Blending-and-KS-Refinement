"""
This module loads high-resolution (HR) and low-resolution (LR) image pairs, preprocesses them by converting 
to Y channel from YCrCb color space and resizing, and then saves the processed images into numpy arrays.
The mapping between HR and LR images is retrieved from a JSON file, and the processed images are stored in a 
compressed .npz file.
"""

import json
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
from skimage.color import rgb2ycbcr
import os

def preprocess_image(image_path: str, target_shape=(128, 128)):
    """
    Preprocess an image by loading, converting it to YCrCb color space, and resizing the Y channel.
    
    Args:
        image_path (str): File path of the image.
        target_shape (tuple, optional): Target dimensions (height, width) for resizing. Defaults to (128, 128).
        
    Returns:
        np.ndarray: The processed Y channel of the image.
    """
    image = load_img(image_path, color_mode='rgb')
    image_array = img_to_array(image)
    ycbcr_image = rgb2ycbcr(image_array / 255.0)
    y_channel = resize(ycbcr_image[:, :, 0], target_shape)
    return y_channel

def prepare_data(json_file_path: str, output_file_path: str, limit: int = None, progress_bar=None):
    """
    Prepares the dataset by mapping high-resolution images to their corresponding low-resolution images.
    Each HR image is preprocessed and its LR counterparts are combined along the channel axis. The results 
    are then saved in a .npz file.
    
    Args:
        json_file_path (str): Path to the JSON file containing HR-LR image mapping.
        output_file_path (str): File path to save the processed data (.npz file).
        limit (int, optional): Limit on the number of HR images to process. Defaults to None (process all).
        progress_bar: Optional global progress bar (e.g., a tqdm instance) to update.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
      
    input_images = []
    target_images = []
      
    items = list(data.items())
    if limit is not None:
        items = items[:limit]
  
    for hr_image, lr_images in items:
        hr_image_path = f'data/high_resolution_images/HR_{hr_image}'
        target_images.append(preprocess_image(hr_image_path))
  
        lr_channel_images = []
        for lr_image in lr_images:
            lr_image_path = f'data/low_resolution_images/{lr_image}'
            lr_channel_images.append(preprocess_image(lr_image_path))
        input_images.append(np.stack(lr_channel_images, axis=-1))
  
        # print(f"Processed HR image: {hr_image}")
        if progress_bar:
            progress_bar.update(1)
  
    input_images = np.array(input_images)
    target_images = np.array(target_images)
  
    print(f"Saving processed data to {output_file_path}...")
    np.savez(output_file_path, input_images=input_images, target_images=target_images)
    print("Data preparation and saving complete.")

if __name__ == '__main__':
    json_file_path = 'data/hr_lr_pairs.json'
    output_file_path = 'logs/processed_data.npz'
    limit = None  # Set to None to process the entire dataset
    prepare_data(json_file_path, output_file_path, limit)
