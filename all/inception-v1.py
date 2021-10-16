"""
    Implementation of Inception-v1 Model Architecture
    Paper: https://arxiv.org/pdf/1409.4842.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, AveragePooling2D, MaxPooling2D,
                                     Input, Concatenate)
