"""
    Implementation of Inception-v1 Model Architecture
    Paper: https://arxiv.org/pdf/1409.4842.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, AveragePooling2D, MaxPooling2D,
                                     Input, Concatenate)


# Defining the input layer
input_layer = Input(shape=(224, 224, 3))

# Defining stem block
conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation=relu, padding="same")(input_layer)
mp_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_1)
conv_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding="same")(mp_1)
conv_3 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(conv_2)
mp_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_3)
