"""
    Implementation of Inception-v1 Model Architecture
    Paper: https://arxiv.org/pdf/1409.4842.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, AveragePooling2D, MaxPooling2D,
                                     Input, Concatenate)


def build_inception_module(pl, nf_11: int, nf_33_r: int, nf_55_r: int, nf_33: int,
                           nf_55: int, nf_pp: int) -> Concatenate:
    """
    This function builds inception module which is mentioned in the model diagram
    :param pl: previous layer
    :param nf_11: number of filters in 1x1 convolutional layer
    :param nf_33_r: number of filters (dimension reduction)
    :param nf_55_r: number of filters (dimension reduction)
    :param nf_33: number of filters in 3x3 convolutional layer
    :param nf_55: number of filters in 5x5 convolutional layer
    :param nf_pp: number of filters in pool proj layer
    :return: concatenated layer
    """
    conv_11 = Conv2D(filters=nf_11, kernel_size=(1, 1), activation=relu, padding="same")(pl)
    conv_33_r = Conv2D(filters=nf_33_r, kernel_size=(1, 1), activation=relu, padding="same")(pl)
    conv_55_r = Conv2D(filters=nf_55_r, kernel_size=(1, 1), activation=relu, padding="same")(pl)
    mp = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(pl)
    conv_33 = Conv2D(filters=nf_33, kernel_size=(3, 3), activation=relu, padding="same")(conv_33_r)
    conv_55 = Conv2D(filters=nf_55, kernel_size=(5, 5), activation=relu, padding="same")(conv_55_r)
    pool_proj = Conv2D(filters=nf_pp, kernel_size=(1, 1), activation=relu, padding="same")(mp)
    concat = Concatenate()([conv_11, conv_33, conv_55, pool_proj])
    return concat


# Defining the input layer
input_layer = Input(shape=(224, 224, 3))

# Defining stem block
conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation=relu, padding="same")(input_layer)
mp_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_1)
conv_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding="same")(mp_1)
conv_3 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(conv_2)
mp_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_3)

# Defining inception module number 1
inc_1 = build_inception_module(mp_2, 64, 96, 16, 128, 32, 32)
