""""
    Convolutional Neural Network with Parallel Layers
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Dense, BatchNormalization, Concatenate,
                                     Flatten, MaxPooling2D, AveragePooling2D, Input)


# Defining input layer
input_ = Input(shape=(224, 224, 3))

# Defining the first parallel layer
in_1 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(input_)
conv_1 = BatchNormalization()(in_1)
conv_1 = AveragePooling2D(pool_size=(2, 2), strides=(3, 3))(conv_1)
