"""
    Implementation of AlexNet Model Architecture
    Paper: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
