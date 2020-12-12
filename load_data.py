import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load Fashion-MNIST Dataset
def load_fashion_mnist():
	num_classes = 10
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	y_train_cat = to_categorical(y_train, num_classes)
	y_test_cat = to_categorical(y_test, num_classes)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	y_train = y_train.astype('float32')
	y_test = y_test.astype('float32')
	x_train /= 255.0
	x_test /= 255.0
	
	return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)
