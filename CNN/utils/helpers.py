import numpy as np
import math
from scipy import ndimage

def augment(mnist):
	new_shape = np.array(list(mnist.x_train.shape))
	augmented_train_datas = np.zeros(new_shape)
	augmented_train_labels = np.zeros([mnist.x_train.shape[0]], dtype=int) 
	for i in range(mnist.x_train.shape[0]):
		img = np.array(list(mnist.x_train[i, 0, :, :]))
		background_value = np.median(img)

		# Rotate
		rotated_img = ndimage.rotate(img, np.random.randint(-30, 30), reshape=False, cval=background_value)

		# Shift
		shifted_img = ndimage.shift(rotated_img, np.random.randint(-2, 2, 2), cval=background_value)

		if(np.random.randint(2) == 0):
			augmented_train_datas[i, 0, :, :] = shifted_img
		else:
			augmented_train_datas[i, 0, :, :] = rotated_img

		augmented_train_labels[i] = int(mnist.y_train[i])

	mnist.x_train = np.append(mnist.x_train, augmented_train_datas, axis=0)

	mnist.y_train = np.append(mnist.y_train, augmented_train_labels, axis=0)

	mnist.x_train, mnist.y_train = shuffle_two_arrays(mnist.x_train, mnist.y_train)
	
	new_shape = np.array(list(mnist.x_test.shape))
	augmented_test_datas = np.zeros(new_shape)
	augmented_test_labels = np.zeros([mnist.x_test.shape[0]], dtype=int) 
	for i in range(mnist.x_test.shape[0]):
		img = np.array(list(mnist.x_test[i, 0, :, :]))
		background_value = np.median(img)

		# Rotate
		rotated_img = ndimage.rotate(img, 15, reshape=False, cval=background_value)

		# Shift
		shifted_img = ndimage.shift(rotated_img, np.array([-1, 1]), cval=background_value)

		if(np.random.randint(2) == 0):
			augmented_test_datas[i, 0, :, :] = shifted_img
		else:
			augmented_test_datas[i, 0, :, :] = rotated_img

		augmented_test_labels[i] = int(mnist.y_test[i])

	mnist.x_test = np.append(mnist.x_test, augmented_test_datas, axis=0)

	mnist.y_test = np.append(mnist.y_test, augmented_test_labels, axis=0)	

	mnist.x_test, mnist.y_test = shuffle_two_arrays(mnist.x_test, mnist.y_test)

def shuffle_two_arrays(x, y):
	assert len(x) == len(y)
	shuffled_x = np.empty(x.shape, dtype=x.dtype)
	shuffled_y = np.empty(y.shape, dtype=y.dtype)
	permutation = np.random.permutation(len(x))
	for old_index, new_index in enumerate(permutation):
		shuffled_x[new_index] = x[old_index]
		shuffled_y[new_index] = y[old_index]

	return shuffled_x, shuffled_y

def transform_bias(bias, sz):
	bias = list(map(lambda x: [x] * sz, bias))
	return np.array(bias)

def process_padding(x, padding):
	padded_x = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2*padding, x.shape[3] + 2*padding))
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			padded_x[i, j] = np.pad(x[i, j], int(padding), 'constant')

	return padded_x