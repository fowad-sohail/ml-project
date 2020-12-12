import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy
import neural_structured_learning as nsl
import numpy as np

# Train basic model
# * Labels must be categorical
def train(model, epochs, train, test):
	model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
	model.fit(x=train[0], y=train[1], epochs=epochs, verbose=1)
	model.evaluate(x=test[0], y=test[1])
	
	return model
	
# Train model with adversarial regularization
# * Labels should not be categorical
def train_regularize(model, epochs, adv_mult, adv_step, train, test):
	adv_config = nsl.configs.make_adv_reg_config(multiplier=adv_mult, adv_step_size=adv_step)
	adv_model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config)
	adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	adv_model.fit({'feature': train[0], 'label': train[1]}, epochs=epochs, verbose=1)
	adv_model.evaluate({'feature': test[0], 'label': test[1]})
	
	return adv_model
