import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Import the main inference function from model_prediction.py (in root directory)
from model.model_prediction import Main

def compute_sharpness(image):
    """
    Computes the sharpness of an image using the variance of the Laplacian.
    
    Args:
        image (np.ndarray): Input image in BGR format.
    
    Returns:
        float: Sharpness score (higher means sharper).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness

def resize_for_display(image, max_width=500):
    """
    Resizes the image for display if its width exceeds max_width.
    
    Args:
        image (np.ndarray): Input image.
        max_width (int): Maximum width for display.
    
    Returns:
        np.ndarray: Resized image for display.
    """
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

# --- Set Page Config and Inject Custom CSS ---
st.set_page_config(page_title="NL2ECF-SRCNN Super Resolution", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
        font-size: 14px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
    }
    h1 {
        text-align: center;
        color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Instructions ---
st.sidebar.title("How It Works")
st.sidebar.info(
    """
    **Steps:**
    
    1. **Upload Image:** Choose a low-resolution image (JPG, JPEG, PNG).
    2. **Enhance Image:** Click the 'Enhance Image' button to run super resolution.
    3. **View Results:** Compare the original and enhanced images side by side.
    
    **Note:** 
    - The sharpness score is computed using the variance of the Laplacian.
    - A higher score generally indicates more detailed edge information.
    """
)

# --- Main App Content ---
st.title("NL2ECF-SRCNN Super Resolution App")
st.markdown(
    """
    Welcome to the NL2ECF-SRCNN Super Resolution App!  
    Enhance your low-resolution images with our advanced model using **Vibrancy-Weighted Blending** and **Kernel Sharpening Refinement**.
    Upload your image below and see a detailed before-and-after comparison.
    """
)

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    
    # Read the image using OpenCV
    image = cv2.imread(tfile.name)
    
    if image is None:
        st.error("Error loading the image. Please try another file.")
    else:
        # Input checks
        height, width, channels = image.shape
        if channels != 3:
            st.warning("The uploaded image may not be a valid color image (expected 3 channels).")
        if height < 64 or width < 64:
            st.warning("The image resolution is very low; the result might not be optimal.")
        
        # Resize for display
        image_for_display = resize_for_display(image)
        
        # Show a preview in an expander
        with st.expander("Preview Uploaded Image"):
            st.image(cv2.cvtColor(image_for_display, cv2.COLOR_BGR2RGB), caption="Original Low-Resolution Image", use_container_width=True)
        
        # --- Run Super Resolution ---
        if st.button("Enhance Image"):
            with st.spinner("Processing image, please wait..."):
                # Run model inference
                output_image = Main(InputImagePath=tfile.name, Model_path="logs/nl2ecf_srcnn_model.h5")
            
            # Compute sharpness scores
            sharpness_lr = compute_sharpness(image)
            sharpness_sr = compute_sharpness(output_image)
            
            # Resize output for display
            output_for_display = resize_for_display(output_image)
            
            st.markdown("### Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image_for_display, cv2.COLOR_BGR2RGB), 
                         caption=f"Low-Resolution Input\nSharpness: {sharpness_lr:.2f}", 
                         use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(output_for_display, cv2.COLOR_BGR2RGB), 
                         caption=f"Enhanced Output\nSharpness: {sharpness_sr:.2f}", 
                         use_container_width=True)
            
            st.success("Image enhancement complete!")
            st.markdown(
                """
                **Footnote:** The sharpness metric is computed using the variance of the Laplacian.  
                A higher sharpness score indicates more detailed edge information, which is often perceived as an improvement.
                """
            )
    
    # Clean up temporary file
    os.remove(tfile.name)
