"""
    Implementation of AlexNet Model Architecture
    Paper: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

model = Sequential(name="AlexNet")
model.add(Input(shape=(224, 224, 3)))
model.add(Conv2D(filters=96, kernel_size=(11, 11), activation=relu))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation=relu))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(filters=384, kernel_size=(3, 3), activation=relu))
model.add(Conv2D(filters=384, kernel_size=(3, 3), activation=relu))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation=relu))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(units=4096, activation=relu))
model.add(Dense(units=4096, activation=relu))
model.add(Dense(units=1000, activation=softmax))


if __name__ == '__main__':
    model.summary()
