"""
    Implementation of LeNet-5 Model Architecture
    Paper: https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@gitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh, softmax
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dense, Flatten


# Defining the input layer
input_layer = Input(shape=(32, 32, 1))

conv_1 = Conv2D(filters=6, kernel_size=(5, 5), activation=tanh)(input_layer)
first_pool = AveragePooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(filters=16, kernel_size=(5, 5), activation=tanh)(first_pool)
second_pool = AveragePooling2D(pool_size=(2, 2))(conv_2)
flat = Flatten()(second_pool)
d_1 = Dense(units=120, activation=tanh)(flat)
d_2 = Dense(units=84, activation=tanh)(d_1)
output_layer = Dense(units=10, activation=softmax)(d_2)

lenet_5 = Model(inputs=[input_layer], outputs=[output_layer], name="LeNet-5")


if __name__ == '__main__':
    lenet_5.summary()
