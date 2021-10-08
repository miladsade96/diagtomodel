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

first_pool = MaxPooling2D(pool_size=(3, 3), strides=2)(first_add)
# Defining convolutional layers up to second addition
conv_4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(first_pool)
conv_5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding="same")(first_pool)
conv_6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(conv_4)
second_add = Add()([conv_5, conv_6])

second_pool = MaxPooling2D(pool_size=(3, 3), strides=2)(second_add)
# Defining convolutional layers up to third addition
conv_7 = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(second_pool)
conv_8 = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding="same")(second_pool)
conv_9 = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(conv_7)
third_add = Add()([conv_8, conv_9])

third_pool = MaxPooling2D(pool_size=(3, 3), strides=2)(third_add)
# Defining convolutional layers up to fourth addition
conv_10 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(third_pool)
conv_11 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding="same")(third_pool)
conv_12 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(conv_10)
fourth_add = Add()([conv_11, conv_12])

fourth_pool = MaxPooling2D(pool_size=(3, 3), strides=2)(fourth_add)
# Defining convolutional layers up to fifth addition
conv_13 = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(fourth_pool)
conv_14 = Conv2D(filters=80, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding="same")(fourth_pool)
conv_15 = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same")(conv_13)
fifth_add = Add()([conv_14, conv_15])

fifth_pool = MaxPooling2D(pool_size=(3, 3), strides=2)(fifth_add)
gap = GlobalAvgPool2D()(fifth_pool)
flat = Flatten()(gap)
output_layer = Dense(units=2, activation=softmax)(flat)

# Creating the model
mla_net = Model(inputs=[input_layer], outputs=[output_layer])


if __name__ == '__main__':
    mla_net.summary()
