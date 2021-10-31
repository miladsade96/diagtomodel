"""
    An implementation of the Xception model architecture.
    Paper: https://arxiv.org/abs/1610.02357
    Author: Prakhar Srivastava - @prkhrsrvstv1
"""
from typing import Any

from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input,
                                     MaxPool2D, SeparableConv2D)
from tensorflow.keras.models import Model


def convolutional_unit(conv_inputs, num_filters, kernel_size, strides=(1, 1), pre_activation=False,
                       post_activation=True, conv_layer: Any = Conv2D):
    if pre_activation:
        conv_inputs = Activation("relu")(conv_inputs)
    conv_outputs = conv_layer(filters=num_filters, kernel_size=kernel_size, strides=strides, padding="same",
                              use_bias=False)(conv_inputs)
    conv_outputs = BatchNormalization()(conv_outputs)
    if post_activation:
        conv_outputs = Activation("relu")(conv_outputs)
    return conv_outputs


def separable_convolutional_unit(sep_conv_inputs, num_filters, pre_activation=True, post_activation=False):
    return convolutional_unit(sep_conv_inputs, num_filters, (3, 3), pre_activation=pre_activation,
                              post_activation=post_activation, conv_layer=SeparableConv2D)


def entry_flow(entry_inputs):
    # Block 2 (Red)
    entry_outputs = convolutional_unit(entry_inputs, 32, (3, 3), (2, 2))
    entry_outputs = convolutional_unit(entry_outputs, 64, (3, 3))

    # Block 3 - Conv A (Yellow)
    for num_filters in [128, 256, 728]:
        res = convolutional_unit(entry_outputs, num_filters, (1, 1), (2, 2), post_activation=False)
        for _ in range(2):
            entry_outputs = separable_convolutional_unit(entry_outputs, num_filters)
        entry_outputs = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(entry_outputs)
        entry_outputs = Add()([res, entry_outputs])
    return entry_outputs


def middle_flow(middle_inputs):
    # Block 4 - Conv B (Green)
    middle_outputs = middle_inputs
    for _ in range(8):
        res = middle_outputs
        for _ in range(3):
            middle_outputs = separable_convolutional_unit(middle_outputs, 728)
        middle_outputs = Add()([res, middle_outputs])
    return middle_outputs


def exit_flow(exit_inputs):
    # Block 5 - Conv C (Orange)
    res = convolutional_unit(exit_inputs, 1024, (1, 1), (2, 2), post_activation=False)
    exit_outputs = exit_inputs
    for num_filters in [728, 1024]:
        exit_outputs = separable_convolutional_unit(exit_outputs, num_filters)
    exit_outputs = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(exit_outputs)
    exit_outputs = Add()([res, exit_outputs])
    for num_filters in [1536, 2048]:
        exit_outputs = separable_convolutional_unit(exit_outputs, num_filters, pre_activation=False,
                                                    post_activation=True)

    # Block 6 Global Average Pool (Gray)
    exit_outputs = GlobalAveragePooling2D()(exit_outputs)

    # Optional fully-connected layer(s) (Blue)
    exit_outputs = Dense(units=2048)(exit_outputs)
    exit_outputs = Activation("relu")(exit_outputs)

    # Logistic regression (Blue)
    exit_outputs = Dense(units=1000)(exit_outputs)
    exit_outputs = Activation("softmax")(exit_outputs)

    return exit_outputs


# Defining the input layer with images of shape 299 x 299 x 3
model_inputs = Input(shape=(299, 299, 3))
# Pass through the entry flow
intermediate_outputs = entry_flow(model_inputs)
# Pass through the middle flow
intermediate_outputs = middle_flow(intermediate_outputs)
# Pass through the exit flow
model_outputs = exit_flow(intermediate_outputs)

model = Model(inputs=model_inputs, outputs=model_outputs, name="Xception")

if __name__ == '__main__':
    model.summary()
