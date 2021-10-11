"""
    Implementation of LeNet-5 Model Architecture
    Paper: https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@gitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh, softmax
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dense, Flatten
