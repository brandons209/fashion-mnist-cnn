#general imports
import numpy as np
import time
import pickle

#data set imports
from keras.datasets import fashion_mnist

#preposses data
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

#model imports
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

#training
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import optimizers as opt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#one-hot encode labels
train_labels, test_labels = to_categorical(train_labels, num_classes=10), to_categorical(test_labels, num_classes=10)
valid_split = 0.1

#augment data, validation split gives subset parameter for flow function
#augmented_data_gen = ImageDataGenerator(rotation_range=270, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, validation_split=valid_split)

#normalize pixel values from [0, 255] to [0, 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

print("Dataset Stats")
print("Training images shape: {}".format(train_images.shape))
print("There are {} training images, {} testing images, and the validation split is {}% of training data, which is {} images.".format(len(train_images), len(test_images), valid_split*100, len(train_images)*valid_split))

#build model
cnn = Sequential()

cnn.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu', input_shape=train_images.shape[1:]))
cnn.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(strides=2, padding='same'))
cnn.add(Dropout(0.3))

cnn.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(strides=2, padding='same'))
cnn.add(Dropout(0.4))

cnn.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
cnn.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
cnn.add(Dropout(0.4))

cnn.add(Flatten())
cnn.add(Dense(300, activation='relu'))
cnn.add(Dropout(0.3))
cnn.add(Dense(100, activation='relu'))
cnn.add(Dense(10, activation='softmax'))

cnn.summary()
input("Press enter to start training model...")
#hyperparameters
epochs = 2
batch_size = 64
learn_rate = 0.01

cnn.compile(loss='categorical_crossentropy', optimizer=opt.RMSProp(lr=learn_rate), metrics=['accuracy', 'top_k_categorical_accuracy'])

weight_save_path = 'fashion.mnist.best.weights.hdf5'
checkpointer = ModelCheckpoint(filepath=weight_save_path, verbose=1, save_best_only=True)

start_time = time.strftime("%a_%b_%d_%Y_%H:%M", time.localtime())
ten_board = TensorBoard(log_dir='tensorboard_logs/{}_fashion_mnist'.format(start_time), write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1*10**(-11))
#TODO add early stop callback if needed

#train model
time_to_train = time.time()
#history = cnn.fit_generator(augmented_data_gen.flow(train_images, train_labels, batch_size=batch_size, subset='training'), steps_per_epoch=train_images.shape[0] // batch_size, epochs=epochs, verbose=1, callbacks=[checkpointer, ten_board, reduce_lr], validation_data=augmented_data_gen.flow(train_images, train_labels, batch_size=batch_size, subset='validation'), validation_steps=(train_images.shape[0]*valid_split) // batch_size, use_multiprocessing=True)
history = cnn.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer, ten_board, reduce_lr], validation_split=valid_split)
time_to_train = time.time() - time_to_train
print("Model took {:.0f} minutes to train".format(time_to_train/60))
input("Press enter to test and save model...")

#test model
cnn.load_weights(weight_save_path)
testing_results = cnn.evaluate(test_images, test_labels)
print("Testing Loss: {:.4f}, Testing Accuracy: {:.2f}%, Top 5 Accuracy: {:.2f}%.\n".format(testing_results[0], testing_results[1]*100, testing_results[2]*100))

model_save_path = "saved_models/"+start_time+"_fashion_mnist.h5"
print("Saving model to {}".format(model_save_path))
cnn.save(model_save_path)

history_save_path = "{}_cnn_history.pkl".format(start_time)
print("Saving history to {}.".format(history_save_path))
with open(history_save_path, 'wb') as f:
    pickle.dump(history.history, f)
