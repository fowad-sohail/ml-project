import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load Fashion-MNIST Dataset
input_shape = (28, 28, 1)
num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("Train/test datset shapes")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Define model
def lenet(nb_classes):
    input = Input(shape = (28, 28, 1))
    conv1 = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same")(input)
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)
    conv2 = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(pool1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
    flat = Flatten()(pool2)
    dense1 = Dense(120, activation='relu')(flat)
    dense2 = Dense(84, activation='relu')(dense1)
    out = Dense(nb_classes, activation='softmax')(dense2)
    model = Model(inputs=input, outputs=out)
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    
    return model

model = lenet(num_classes)
model.summary()

# Train and save model
model.fit(x_train, y=y_train, epochs=20, validation_data=(x_test, y_test))
model.save('lenet_noreg_20epoch')

