import tensorflow as tf

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
import numpy as np
from cleverhans.future.tf2.attacks import fast_gradient_method

# Load Fashion-MNIST Dataset
num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_test = x_test[:, :, :, np.newaxis].astype('float32')
x_test /= 255
y_test = to_categorical(y_test, num_classes)

# Load Trained Model
model = tf.keras.models.load_model('lenet_v2')

# Attack
adversarial_x_test = fast_gradient_method(model, x_test, 0.3, np.inf)

print(model.evaluate(x=adversarial_x_test, y=y_test, verbose=1))

