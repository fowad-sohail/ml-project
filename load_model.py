from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Input, Reshape

# Basic LeNet model
def lenet_base():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model
