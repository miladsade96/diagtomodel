""""
    Convolutional Neural Network with Multiple Layer Additions
    Author: Milad Sadeghi Dm - EverLookNeverSee@GitHub.com
"""


from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Conv2D, Dense, Input, GlobalAvgPool2D, Flatten, MaxPooling2D, Add


# Defining the input layer
input_layer = Input(shape=(720, 720, 3))

# Defining convolutional layers up to first addition
conv_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation=relu, padding="same")(input_layer)
conv_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(4, 4), activation=relu, padding="same")(input_layer)
conv_3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation=relu, padding="same")(conv_1)
first_add = Add()([conv_2, conv_3])
