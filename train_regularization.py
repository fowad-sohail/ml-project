import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
import numpy as np
from absl import app, flags
from cleverhans.future.tf2.attacks import projected_gradient_descent, fast_gradient_method
import neural_structured_learning as nsl

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

FLAGS = flags.FLAGS

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
batch_size = 32

# Add a new axis
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

input_shape = (1, 28, 28, 1)
num_classes = 10

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def lenet_sequential(nb_classes, input_shape):
	model = Sequential()
	model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same", input_shape=input_shape))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(Flatten())
	model.add(Dense(120, activation='relu'))
	model.add(Dense(84, activation='relu'))
	model.add(Dense(nb_classes, activation='softmax'))
	adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
	adv_model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config)
	adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#	adv_model.build(input_shape)
#	adv_model(np.random.uniform(size=input_shape))
	
	return adv_model

def lenet(nb_classes, input_shape):
	input = Input(shape = input_shape)
	conv1 = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same")(input)
	pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)
	conv2 = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(pool1)
	pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
	flat = Flatten()(pool2)
	dense1 = Dense(120, activation='relu')(flat)
	dense2 = Dense(84, activation='relu')(dense1)
	out = Dense(nb_classes, activation='softmax')(dense2)
	model = Model(inputs=input, outputs=out)
	adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
	adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)
	adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	adv_model.build(input_shape)
	
	return adv_model

# LeNet-5 model
#class LeNet(Sequential):
#    def __init__(self, input_shape, nb_classes):
#        super().__init__()
#
#        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
#        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
#        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#        self.add(Flatten())
#        self.add(Dense(120, activation='relu'))
#        self.add(Dense(84, activation='relu'))
#        self.add(Dense(nb_classes, activation='softmax'))
#
#        self.compile(optimizer='adam',
#                    loss=categorical_crossentropy,
#                    metrics=['accuracy'])

model = lenet_sequential(num_classes, input_shape)
#model.summary()

train_data = tf.data.Dataset.from_tensor_slices(
    {'input': [x_train], 'label': [y_train]}).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices(
    {'input': [x_test], 'label': [y_test]}).batch(batch_size)
val_steps = x_test.shape[0] / batch_size

model.fit(train_data,
          epochs=2,
          validation_data=val_data, validation_steps=val_steps, verbose=1)
          
model.save('lenet_regularization_epoch2')


# Attack
#sess = tf.Session()
#keras.backend.set_session(sess)
#
#wrap = KerasModelWrapper(model)
#fgsm = FastGradientMethod(wrap, sess=sess)
#fgsm_params = {'eps': 0.3,
#                'clip_min': 0.,
#                'clip_max': 1.}
#adversarial_x_test = fast_gradient_method(model, x_test, 0.3, np.inf)
#
#print(x_test.shape)
#print(adversarial_x_test.shape)

#adversarial_x_test = fgsm.generate(x_test, **fgsm_params)
#preds_adv = model(adversarial_x_test)

# model.evaluate(x=adversarial_x_test, y=y_test, verbose=1)
#acc = model_eval(sess, x_train, y_train, preds_adv, x_test, y_test)
