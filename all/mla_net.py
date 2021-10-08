""""
    Convolutional Neural Network with Multiple Layer Additions
    Author: Milad Sadeghi Dm - EverLookNeverSee@GitHub.com
"""


from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Conv2D, Dense, Input, GlobalAvgPool2D, Flatten, MaxPooling2D, Add


# Defining the input layer
input_layer = Input(shape=(720, 720, 3))
