"""
This module handles inference for the NL2ECF-SRCNN model. It includes functions for:
    - Loading and preprocessing input images.
    - Running the NL2ECF-SRCNN model to predict the high-resolution luminance.
    - Postprocessing the prediction using Vibrancy-Weighted Blending and Kernel Sharpening Refinement 
      to produce the final color super-resolved image.
    - Displaying images side by side for comparison.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from model.model_architecture import NL2ECF_SRCNN_model, original_SRCNN_Model


def modcrop(img, scale):
    """
    Crops the image so that its dimensions are multiples of the scale factor.
    
    Args:
        img (np.ndarray): Input image.
        scale (int): Scale factor.
    
    Returns:
        np.ndarray: Cropped image.
    """
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    print(f"Image cropped to size {sz}")
    return img


def display_side_by_side(original_image, downscaled_image, super_resolution_image):
    """
    Displays the original, downscaled, and super-resolved images side by side.
    
    Args:
        original_image (np.ndarray): Original high-resolution image.
        downscaled_image (np.ndarray): Low-resolution input image.
        super_resolution_image (np.ndarray): Final super-resolved image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    downscaled_image = cv2.cvtColor(downscaled_image, cv2.COLOR_BGR2RGB)
    super_resolution_image = cv2.cvtColor(super_resolution_image, cv2.COLOR_BGR2RGB)

    axes[0].imshow(original_image)
    axes[0].set_title('Original High Resolution')
    axes[0].axis('off')

    axes[1].imshow(downscaled_image)
    axes[1].set_title('Low Resolution Input')
    axes[1].axis('off')

    axes[2].imshow(super_resolution_image)
    axes[2].set_title('Super-Resolved Output')
    axes[2].axis('off')

    plt.show()
    print("Displayed images side by side.")


def load_model(weights="nl2ecf_srcnn_model.h5"):
    """
    Loads the NL2ECF-SRCNN model and its trained weights.
    
    Args:
        weights (str): Path to the weights file.
    
    Returns:
        model: Loaded NL2ECF-SRCNN model.
    """
    model = NL2ECF_SRCNN_model()
    # Uncomment the following line to use the original model for testing.
    # model = original_SRCNN_Model()
    model.load_weights(weights)
    print(f"Model loaded successfully with weights: {weights}")
    return model


def load_image(image_path):
    """
    Loads an image from the given path.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        np.ndarray: Loaded image.
    """
    image = cv2.imread(image_path)
    return image


def preprocess_image(image, size=(128, 128)):
    """
    Preprocesses an image by converting it to YCbCr, resizing the Y channel, and normalizing it.
    
    Args:
        image (np.ndarray): Input image in BGR format.
        size (tuple): Target size for the Y channel.
    
    Returns:
        tuple: Preprocessed image ready for model input and the full YCbCr image.
    """
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = ycbcr_image[:, :, 0]
    Y_resized = cv2.resize(Y, size, interpolation=cv2.INTER_CUBIC)
    Y_normalized = Y_resized.astype(float) / 255.0
    Y_normalized = np.expand_dims(Y_normalized, axis=0)
    Y_normalized = np.expand_dims(Y_normalized, axis=-1)
    return Y_normalized, ycbcr_image


def model_predictions(model, preprocessed_image):
    """
    Runs the NL2ECF-SRCNN model to predict the high-resolution Y channel.
    
    Args:
        model: Loaded NL2ECF-SRCNN model.
        preprocessed_image (np.ndarray): Preprocessed input image.
    
    Returns:
        np.ndarray: Predicted image.
    """
    predicted_image = model.predict(preprocessed_image, batch_size=1)
    print("Model prediction completed.")
    return predicted_image


def nl2ecf_postprocess(predicted_image, ycbcr_image, vibrancy_weight=0.2):
    """
    Postprocesses the model output using Vibrancy-Weighted Blending and Kernel Sharpening Refinement.
    This function fuses the predicted Y channel with the original YCbCr channels to produce the final 
    BGR image with enhanced brightness, vibrancy, and sharpness.

    Args:
        predicted_image (np.ndarray): Predicted image from the model.
        ycbcr_image (np.ndarray): Original YCbCr image.
        vibrancy_weight (float): Weight to balance the contribution of the predicted Y channel (default: 0.2).

    Returns:
        np.ndarray: Final postprocessed image in BGR format.
    """
    # Recover and scale the predicted Y channel.
    predicted_y = predicted_image[0, :, :, 0] * 255
    predicted_y = np.clip(predicted_y, 0, 255)

    # Resize predicted Y to match original dimensions.
    original_height, original_width = ycbcr_image.shape[:2]
    predicted_y_resized = cv2.resize(predicted_y, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)

    # Blend the original and predicted Y channels.
    original_y = ycbcr_image[:, :, 0]
    final_y = (1 - vibrancy_weight) * original_y + vibrancy_weight * predicted_y_resized

    # Replace the Y channel in the original YCbCr image.
    final_ycbcr = ycbcr_image.copy()
    final_ycbcr[:, :, 0] = np.clip(final_y, 0, 255).astype(np.uint8)

    # Convert back to BGR.
    final_bgr = cv2.cvtColor(final_ycbcr, cv2.COLOR_YCrCb2BGR)

    # Apply kernel sharpening refinement.
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    final_bgr = cv2.filter2D(final_bgr, -1, sharpening_kernel)

    return final_bgr


def display_predicted_image(image):
    """
    Displays the super-resolved image in a window.
    
    Args:
        image (np.ndarray): Image to display.
    """
    cv2.imshow("Super-Resolved Image", image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    print("Super-resolved image displayed.")


def compute_sharpness(image):
    """
    Computes the sharpness of an image using the variance of the Laplacian.
    
    Args:
        image (np.ndarray): Input image in BGR format.
    
    Returns:
        float: Sharpness measure (higher values indicate a sharper image).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness


def display_side_by_side_4_UI(downscaled_image, super_resolution_image):
    """
    Displays the low-resolution input and the final super-resolved image side by side,
    along with their computed sharpness values to help indicate improvement.
    
    Args:
        downscaled_image (np.ndarray): Low-resolution image in BGR format.
        super_resolution_image (np.ndarray): Final super-resolved image in BGR format.
    """
    # Compute sharpness metrics
    sharpness_lr = compute_sharpness(downscaled_image)
    sharpness_sr = compute_sharpness(super_resolution_image)
    
    # Convert images to RGB for display using matplotlib
    downscaled_rgb = cv2.cvtColor(downscaled_image, cv2.COLOR_BGR2RGB)
    super_res_rgb = cv2.cvtColor(super_resolution_image, cv2.COLOR_BGR2RGB)
    
    # Create subplots and display images with titles that include the sharpness values
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(downscaled_rgb)
    axes[0].set_title(f'Low Resolution Input\nSharpness: {sharpness_lr:.2f}', fontsize=14)
    # axes[0].axis('off')
    
    axes[1].imshow(super_res_rgb)
    axes[1].set_title(f'Super-Resolved Output\nSharpness: {sharpness_sr:.2f}', fontsize=14)
    # axes[1].axis('off')
    
    plt.subplots_adjust(wspace=0.05)
    plt.show()
    
    print("Displayed images side by side with sharpness metrics.")


def Main(InputImagePath=r"data/low_resolution_images/LR_GB_0001.png", Model_path=r"logs/nl2ecf_srcnn_model.h5"):
    """
    Main function for performing inference using the NL2ECF-SRCNN model:
      1. Loads the model and weights.
      2. Preprocesses the input image.
      3. Predicts the high-resolution Y channel.
      4. Postprocesses the prediction using vibrancy-weighted blending and kernel sharpening.
    
    Args:
        InputImagePath (str): Path to the low-resolution input image.
        Model_path (str): Path to the model weights file.
    
    Returns:
        np.ndarray: Final super-resolved image.
    """
    model = load_model(Model_path)
    input_img = load_image(InputImagePath)
    preprocessed_image, ycbcr_image = preprocess_image(input_img)
    predicted_image = model_predictions(model, preprocessed_image)
    final_image = nl2ecf_postprocess(predicted_image, ycbcr_image)
    return final_image


if __name__ == "__main__":
    # Example usage:
    Model_path = r"logs/nl2ecf_srcnn_model.h5"
    hr_image = cv2.imread('data/high_resolution_images/HR_0757.png')
    lr_image = cv2.imread('data/low_resolution_images/LR_GB_0757.png')
    final_image = Main('data/low_resolution_images/LR_GB_0757.png', Model_path)
    display_side_by_side_4_UI(lr_image, final_image)