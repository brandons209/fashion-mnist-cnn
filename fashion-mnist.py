#general imports
import numpy as np
import time
import pickle

#data set imports
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

#model imports
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

#training
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import optimizers as opt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_labels, test_labels = to_categorical(train_labels, num_classes=10), to_categorical(test_labels, num_classes=10)
valid_split = 0.1

print("Dataset Stats")
print("Training images shape: {}".format(train_images.shape))
print("There are {} training images, {} testing images, and the validation split is {}% of training data".format(len(train_images), len(test_images), valid_split*100))

#build model
cnn = Sequential()
