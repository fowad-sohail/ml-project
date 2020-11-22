import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Input, Reshape
from tensorflow.keras.utils import to_categorical
import numpy as np
import neural_structured_learning as nsl

# Load Fashion-MNIST Dataset
input_shape = (28, 28)
num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

print("Train/test dataset shapes")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def lenet_sequential(nb_classes, input_shape):
	model = Sequential([
		Input(shape=input_shape, name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(nb_classes, activation='softmax')
	])
	adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
	adv_model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config)
	adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	return adv_model

model = lenet_sequential(num_classes, input_shape)

# Train and save model
model.fit({'feature': x_train, 'label': y_train}, epochs=20, verbose=1)
model.evaluate({'feature': x_test, 'label': y_test})
model.save('lenet_reg_20epoch')
