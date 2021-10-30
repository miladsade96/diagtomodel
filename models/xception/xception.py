from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, MaxPool2D, SeparableConv2D
from tensorflow.keras.models import Model

# 299 x 299 x 3 images
inputs = Input(shape=(299, 299, 3))

# Entry flow
outputs = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), use_bias=False)(inputs)
outputs = BatchNormalization()(outputs)
outputs = Activation("relu")(outputs)

outputs = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False)(outputs)
outputs = BatchNormalization()(outputs)
outputs = Activation("relu")(outputs)

for filters in [128, 256, 728]:
    res = Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding="same", use_bias=False)(outputs)
    res = BatchNormalization()(res)
    for _ in range(2):
        outputs = Activation("relu")(outputs)
        outputs = SeparableConv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=False)(outputs)
        outputs = BatchNormalization()(outputs)
    outputs = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(outputs)
    outputs = Add()([res, outputs])

# Middle flow
for _ in range(8):
    res = outputs
    for _ in range(3):
        outputs = Activation("relu")(outputs)
        outputs = SeparableConv2D(filters=728, kernel_size=(3, 3), padding="same", use_bias=False)(outputs)
        outputs = BatchNormalization()(outputs)
    outputs = Add()([res, outputs])

# Exit flow
res = Conv2D(filters=1024, kernel_size=(1, 1), strides=(2, 2), padding="same", use_bias=False)(outputs)
res = BatchNormalization()(res)
for filters in [728, 1024]:
    outputs = Activation("relu")(outputs)
    outputs = SeparableConv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=False)(outputs)
    outputs = BatchNormalization()(outputs)
outputs = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(outputs)
outputs = Add()([res, outputs])

for filters in [1536, 2048]:
    outputs = SeparableConv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=False)(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation("relu")(outputs)

outputs = GlobalAveragePooling2D()(outputs)

# Optional fully-connected layer(s)
outputs = Dense(units=2048)(outputs)
outputs = Activation("relu")(outputs)

# Logistic regression
outputs = Dense(units=1000)(outputs)
outputs = Activation("softmax")(outputs)

model = Model(inputs=inputs, outputs=outputs, name="Xception")

if __name__ == '__main__':
    model.summary()
