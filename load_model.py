from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Input, Reshape

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

# LeNet model with max pooling
def lenet_max():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model
	
# LeNet model with 5 dense layers
def lenet_deep5():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(240, activation='relu'),
		Dense(200, activation='relu'),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model
	
# LeNet model with max pooling and 5 dense layers
def lenet_max_deep5():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(240, activation='relu'),
		Dense(200, activation='relu'),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model

# LeNet model with larger kernels
def lenet_large():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding="same"),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='valid'),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model

# LeNet model with max pooling and larger kernels
def lenet_max_large():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding="same"),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='valid'),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model
	
# LeNet model with 5 dense layers and larger kernels
def lenet_deep5_large():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding="same"),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='valid'),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(240, activation='relu'),
		Dense(200, activation='relu'),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model
	
# LeNet model with max pooling and 5 dense layers and larger kernels
def lenet_max_deep5_large():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(6, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding="same"),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(16, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='valid'),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(240, activation='relu'),
		Dense(200, activation='relu'),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model

# LeNet model with doubled filters
def lenet_filters2():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(12, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
		AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model

# LeNet model with max pooling and doubled filters
def lenet_max_filters2():
	model = Sequential([
		Input(shape=(28, 28), name='feature'),
		Reshape((28, 28, 1)),
		Conv2D(12, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
		Flatten(),
		Dense(120, activation='relu'),
		Dense(84, activation='relu'),
		Dense(10, activation='softmax')
	])

	return model
