"""
    Implementation of Inception-v1 Model Architecture
    Paper: https://arxiv.org/pdf/1409.4842.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, AveragePooling2D, MaxPooling2D,
                                     Input, Concatenate, Dropout, GlobalAveragePooling2D)


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
# Defining inception module number 2
inc_2 = build_inception_module(inc_1, 128, 128, 32, 192, 96, 64)

mp_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(inc_2)

# Defining inception module number 3
inc_3 = build_inception_module(mp_3, 192, 96, 16, 208, 48, 64)

# Implementing first output branch
# Average pool
ap_1 = AveragePooling2D(pool_size=(5, 5), strides=3)(inc_3)
conv_4 = Conv2D(filters=128, kernel_size=(1, 1), activation=relu, padding="same")(ap_1)
flat_1 = Flatten()(conv_4)
# fully connected layer
fcl_1 = Dense(units=1024, activation=relu)(flat_1)
do_1 = Dropout(rate=0.7)(fcl_1)
output_1 = Dense(units=10, activation=softmax, name="First Output")(do_1)

# Defining triple inception modules
inc_4 = build_inception_module(inc_3, 160, 112, 24, 224, 64, 64)
inc_5 = build_inception_module(inc_4, 128, 128, 24, 265, 64, 64)
inc_6 = build_inception_module(inc_5, 112, 144, 32, 288, 64, 64)

# Implementing second output branch
# Average pool
ap_2 = AveragePooling2D(pool_size=(5, 5), strides=3)(inc_6)
conv_5 = Conv2D(filters=128, kernel_size=(1, 1), activation=relu, padding="same")(ap_2)
flat_2 = Flatten()(conv_5)
# fully connected layer
fcl_2 = Dense(units=1024, activation=relu)(flat_2)
do_2 = Dropout(rate=0.7)(fcl_2)
output_2 = Dense(units=10, activation=softmax, name="Second Output")(do_2)

# Defining inception module number 7
inc_7 = build_inception_module(inc_6, 256, 160, 32, 320, 128, 128)
# Max pooling
mp_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(inc_7)

# Defining double inception modules
inc_8 = build_inception_module(mp_4, 256, 160, 32, 320, 128, 128)
inc_9 = build_inception_module(inc_8, 384, 192, 48, 384, 128, 128)

# Implementing third output branch
# Global Average Pooling
gap_1 = GlobalAveragePooling2D()(inc_9)
do_3 = Dropout(rate=0.4)(gap_1)
output_3 = Dense(units=10, activation=softmax, name="Third Output")(do_3)
