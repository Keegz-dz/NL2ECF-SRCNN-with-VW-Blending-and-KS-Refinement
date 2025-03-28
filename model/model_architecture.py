"""
This module defines the architectures for the NL2ECF-SRCNN model, which stands for
Non-Linear Luminance Enhancement and Color Fusion SRCNN with Vibrancy-Weighted Blending and 
Kernel Sharpening Refinement. The NL2ECF-SRCNN model is derived from the original SRCNN architecture 
with modifications such as the use of LeakyReLU activations and flexible input dimensions. 

The enhancements (non-linear luminance enhancement and color fusion) are primarily applied during 
postprocessing (see model_prediction.py), while this network focuses on recovering a high-quality Y channel.
It includes:
    - NL2ECF_SRCNN_model: The modified network supporting variable input sizes.
    - original_SRCNN_Model: The original SRCNN model with fixed input size (for reference/testing).
"""

from keras import models
from keras.layers import Conv2D, Input, LeakyReLU
from keras.optimizers import Adam

def NL2ECF_SRCNN_model():
    """
    Builds the NL2ECF-SRCNN model for super-resolution. This model is designed to recover 
    the high-resolution luminance (Y) channel from a low-resolution input while allowing for 
    non-linear enhancements through subsequent postprocessing.

    Key modifications over the original SRCNN:
      - Uses LeakyReLU activations with negative_slope (instead of alpha) to allow for non-linearity.
      - Employs an explicit Input layer to support images of variable dimensions.
      - Designed to work on a single channel input (Y channel from YCbCr).

    Returns:
        model (keras.models.Model): Compiled NL2ECF-SRCNN model.
    """
    # Define model input
    inputs = Input(shape=(None, None, 1))
    
    # First convolution layer with LeakyReLU activation
    x = Conv2D(
        filters=128, 
        kernel_size=(9, 9),
        strides=1,
        padding='same',  # Preserves spatial dimensions
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )(inputs)
    x = LeakyReLU(negative_slope=0.3)(x)
    
    # Second convolution layer with LeakyReLU activation
    x = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )(x)
    x = LeakyReLU(negative_slope=0.3)(x)
    
    # Final convolution layer producing the output Y channel with linear activation
    outputs = Conv2D(
        filters=1,
        kernel_size=(5, 5),
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform'
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    adam = Adam(learning_rate=0.0003)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


"""
    Original SRCNN Attribution:
    The original SRCNN architecture implemented in original_SRCNN_Model is based on the work 
    presented in:
    
        Dong, C., Loy, C.C., He, K., & Tang, X. (2014). Image Super-Resolution Using Deep 
        Convolutional Networks. 
"""
def original_SRCNN_Model():
    """
    Builds the original SRCNN model (fixed input size) for reference or comparison purposes.
    
    Returns:
        model (keras.models.Sequential): Compiled original SRCNN model.
    """
    model = models.Sequential()

    model.add(Conv2D(
        filters=128, 
        kernel_size=(9, 9), 
        kernel_initializer='glorot_uniform',
        padding='valid',
        activation='relu', 
        use_bias=True, 
        input_shape=(64, 64, 1)
    ))
    
    model.add(Conv2D(
        filters=64, 
        kernel_size=(3, 3), 
        kernel_initializer='glorot_uniform',
        padding='same',     
        activation='relu', 
        use_bias=True
    ))
    
    model.add(Conv2D(
        filters=1, 
        kernel_size=(5, 5), 
        kernel_initializer='glorot_uniform',
        padding='valid',
        activation='linear',  
        use_bias=True
    ))

    adam = Adam(learning_rate=0.0003)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model
