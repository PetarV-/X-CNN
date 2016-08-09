from __future__ import print_function
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
import PIL.Image as im

# Will fetch and process CIFAR-10/100, with only p% of the training set returned
# Can also append the remainder to the test set
def get_cifar(p, append_test, use_c10):
	# The raw data, shuffled and split between training and test sets
	(X_train, y_train), (X_test, y_test) = cifar10.load_data() if use_c10 else cifar100.load_data()

	num_samples = X_train.shape[0]
	num_classes = 10 if use_c10 else 100
	
	# Compute how much to retain per class
	cnts = np.full(num_classes, (num_samples // num_classes) * p)

	rem = []
	gather_x = []
	gather_y = []

	for i in range(0, num_samples):
		cur_cls = y_train[i]
		if cnts[cur_cls] > 0:
			cnts[cur_cls] -= 1
		else:
			rem.append(i)
			if append_test:
				gather_x.append(X_train[i])
				gather_y.append(cur_cls)

	if append_test:
		X_test = np.append(X_test, gather_x, axis=0)
		y_test = np.append(y_test, gather_y, axis=0)

	# Remove the computed indices
	X_train = np.delete(X_train, rem, 0)
	y_train = np.delete(y_train, rem, 0)
	X_train_rgb=np.copy(X_train)

	# convert from RBG to YUV
	for i in range(X_train.shape[0]):
		img = im.fromarray(np.transpose(X_train[i]))
		yuv=img.convert('YCbCr')
		X_train[i]=np.transpose(np.array(yuv))

	for i in range(X_test.shape[0]):
		img = im.fromarray(np.transpose(X_test[i]))
		yuv = img.convert('YCbCr')
		X_test[i]=np.transpose(np.array(yuv))

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, num_classes)
	Y_test = np_utils.to_categorical(y_test, num_classes)

	return (X_train, Y_train), (X_test, Y_test), X_train_rgb
