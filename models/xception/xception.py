"""
    An implementation of the Xception model architecture.
    Paper: https://arxiv.org/abs/1610.02357
    Author: Prakhar Srivastava - @prkhrsrvstv1
"""

from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input,
                                     MaxPool2D, SeparableConv2D)
from tensorflow.keras.models import Model
from tensorflow.python.types.core import Tensor


def convolutional_unit(
        conv_inputs: Tensor,
        num_filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int] = (1, 1),
        pre_activation: bool = False,
        post_activation: bool = True,
        conv_layer: str = "Conv2D") -> Tensor:
    """Convolutional Unit

    Passes the input tensor through a convolutional layer, and a batch-normalization layer.
    Also performs any combination of pre/post relu activation, depending on parameters passed.

    Args:
        conv_inputs: Input tensor
        num_filters: Number of filters to use in the convolutional layer
        kernel_size: Kernel size for the convolutional layer
        strides: Strides size for the convolutional layer
        pre_activation: Whether or not to perform activation before the convolutional layer. False by default.
        post_activation: Whether or not to perform activation after the batch-normalization layer. True by default.
        conv_layer: The type of Convolutional layer to use. Should be either Conv2D or SeparableConv2D

    Returns:
        A 4+D Tensor obtained after passing through activation, convolutional, and batch-normalization layers

    """
    if conv_layer not in ["Conv2D", "SeparableConv2D"]:
        raise ValueError(f"conv_layer must be either Conv2D or SeparableConv2D. Found {conv_layer}")

    if pre_activation:
        conv_inputs = Activation("relu")(conv_inputs)
    conv_outputs = conv_inputs

    if conv_layer == "Conv2D":
        conv_outputs = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding="same",
                              use_bias=False)(conv_inputs)
    elif conv_layer == "SeparableConv2D":
        conv_outputs = SeparableConv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding="same",
                                       use_bias=False)(conv_inputs)

    conv_outputs = BatchNormalization()(conv_outputs)

    if post_activation:
        conv_outputs = Activation("relu")(conv_outputs)
    return conv_outputs


def separable_convolutional_unit(
        sep_conv_inputs: Tensor,
        num_filters: int,
        pre_activation: bool = True,
        post_activation: bool = False) -> Tensor:
    """Separable Convolutional Unit

    Uses the Convolutional Unit (`convolutional_unit`) function to pass the input tensor through a Separable
    Convolutional layer. Also performs any combination of pre/post relu activation, depending on parameters passed.

    Args:
        sep_conv_inputs: Input tensor
        num_filters: Number of filters to use in the convolutional layer
        pre_activation: Whether or not to perform activation before the convolutional layer. True by default.
        post_activation: Whether or not to perform activation after the batch-normalization layer. False by default.

    Returns:
        A 4+D Tensor obtained after passing through activation, convolutional, and batch-normalization layers
    """
    return convolutional_unit(sep_conv_inputs, num_filters, (3, 3), pre_activation=pre_activation,
                              post_activation=post_activation, conv_layer="SeparableConv2D")


def entry_flow(entry_inputs: Input) -> Tensor:
    """Entry Flow

    Implements the first of the three broad parts of the model

    Args:
        entry_inputs: Input tensor of shape [*, rows, cols, channels]

    Returns:
        Output tensor of shape [*, new_rows, new_cols, 728]

    """
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


def middle_flow(middle_inputs: Tensor) -> Tensor:
    """Middle Flow

    Implements the second of the three broad parts of the model

    Args:
        middle_inputs: Tensor output generate by the Entry Flow, having shape [*, new_rows, new_cols, 728]

    Returns:
        Output tensor of shape [*, new_rows, new_cols, 728]

    """
    # Block 4 - Conv B (Green)
    middle_outputs = middle_inputs
    for _ in range(8):
        res = middle_outputs
        for _ in range(3):
            middle_outputs = separable_convolutional_unit(middle_outputs, 728)
        middle_outputs = Add()([res, middle_outputs])
    return middle_outputs


def exit_flow(exit_inputs: Tensor) -> Tensor:
    """Exit Flow

    Implements the second of the three broad parts of the model. Includes the optional fully-connected layers,
    and the logistic regression segment of the model.

    Args:
        exit_inputs: Tensor output generated by the Middle Flow segment, having shape [*, new_rows, new_cols, 728]

    Returns:
        Output tensor of shape [*, 1000], representing the classifier output for 1000 classes

    """
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
