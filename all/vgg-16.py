"""
    Implementation of VGG-16 Model Architecture
    Paper: https://arxiv.org/pdf/1409.1556
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""


from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D


# Defining the input layer
input_layer = Input(shape=(224, 224, 3))

# Defining first conv block
conv_1 = Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding="same")(input_layer)
conv_2 = Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding="same")(conv_1)
pooling_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_2)

# Defining second conv block
conv_3 = Conv2D(filters=128, kernel_size=(3, 3), activation=relu, padding="same")(pooling_1)
conv_4 = Conv2D(filters=128, kernel_size=(3, 3), activation=relu, padding="same")(conv_3)
pooling_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_4)

# Defining third conv block
conv_5 = Conv2D(filters=256, kernel_size=(3, 3), activation=relu, padding="same")(pooling_2)
conv_6 = Conv2D(filters=256, kernel_size=(3, 3), activation=relu, padding="same")(conv_5)
conv_7 = Conv2D(filters=256, kernel_size=(3, 3), activation=relu, padding="same")(conv_6)
pooling_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_7)

# Defining fourth cnv block
conv_8 = Conv2D(filters=512, kernel_size=(3, 3), activation=relu, padding="same")(pooling_3)
conv_9 = Conv2D(filters=512, kernel_size=(3, 3), activation=relu, padding="same")(conv_8)
conv_10 = Conv2D(filters=512, kernel_size=(3, 3), activation=relu, padding="same")(conv_9)
pooling_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_10)

# Defining fifth conv block
conv_11 = Conv2D(filters=512, kernel_size=(3, 3), activation=relu, padding="same")(pooling_4)
conv_12 = Conv2D(filters=512, kernel_size=(3, 3), activation=relu, padding="same")(conv_11)
conv_13 = Conv2D(filters=512, kernel_size=(3, 3), activation=relu, padding="same")(conv_12)
pooling_5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv_13)

# Defining fully connected and output layers
flat = Flatten()(pooling_5)
d_1 = Dense(units=4096, activation=relu)(flat)
d_2 = Dense(units=4096, activation=relu)(d_1)
output_layer = Dense(units=1000, activation=softmax)(d_2)

# Model Creation
model = Model(inputs=[input_layer], outputs=[output_layer], name="VGG-16")

if __name__ == '__main__':
    model.summary()
