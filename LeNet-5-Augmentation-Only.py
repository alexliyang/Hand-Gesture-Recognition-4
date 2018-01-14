from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,Activation
import numpy as np

import argparse
import cv2
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

from keras.optimizers import SGD
import scipy.io
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())


# read input data
hand_data = scipy.io.loadmat('hand_data')
hand_data_test = scipy.io.loadmat('hand_data_test')

# reshape the input data to (160,120,3)
# xtrain is the training images, ytrain is the labels
xtrain = hand_data["training"]
xtrain = xtrain.reshape(1500, 3, 160, 120)
xtrain = np.transpose(xtrain,(0,2,3,1))
ytrain = hand_data["train_label"]


# xtest is the testing images (500)
# ytest is the testing labels
xtest = hand_data_test["testing"]
xtest = xtest.reshape(500, 3, 160, 120)
xtest = np.transpose(xtest,(0,2,3,1))
ytest = hand_data_test['test_label']

# normalize to make convergence faster
xtrain = xtrain.astype('float32') / 255.0
xtest = xtest.astype('float32') / 255.0

# Convert 1-dimensional class arrays to 10-dimensional class matrices
ytrain = np_utils.to_categorical(ytrain, 10)
ytest = np_utils.to_categorical(ytest,10)

model = Sequential()

# Architecture of Lenet-5: INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
# Convolution layer 1. Use 32 convolution filters
# Activation function is ReLU
# base experiment, without dropout and data augmentation
model.add(Conv2D(32, (3, 3), border_mode='valid', input_shape=xtrain.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))

# conv + relu + maxpooling
model.add(Conv2D(32, (3, 3), border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))

# fully connected layer
model.add(Flatten())
model.add(Dense(384))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# fully connected layer
model.add(Dense(10))
model.add(Activation('softmax'))

batch_size = 100
epochs = 200

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

if args["load_model"] < 0:
    datagen = ImageDataGenerator(
        zoom_range=0.2,  # randomly zoom into images
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

    history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                                  steps_per_epoch=int(np.ceil(xtrain.shape[0] / float(batch_size))),
                                  epochs=epochs,
                                  validation_data=(xtest, ytest),
                                  workers=4)

    score = model.evaluate(xtest, ytest)
    print(score)

# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

# randomly select a few testing digits
# for i in np.random.choice(np.arange(0, len(ytest)), size=(10,)):
# 	# classify the digit
# 	probs = model.predict(xtest[np.newaxis, i])
# 	prediction = probs.argmax(axis=1)
#
# 	# resize the image from a 28 x 28 image to a 96 x 96 image so we
# 	# can better see it
# 	image = (xtest[i][0] * 255).astype("uint8")
#
# 	cv2.putText(image, str(prediction[0]), (5, 20),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#
# 	# show the image and prediction
# 	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
# 		np.argmax(ytest[i])))
# 	cv2.imshow("Digit", image)
# 	cv2.waitKey(0)

plt.figure(figsize=[10,8])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=16)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('LeNet-5 Loss Curves',fontsize=16)
plt.savefig('loss_lenet5_augmentation.jpg')
plt.show()

plt.figure(figsize=[10,8])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=16)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('LeNet-5 Accuracy Curves',fontsize=16)
plt.savefig('accuracy_lenet5_augmentation.jpg')
plt.show()