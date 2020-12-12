from tensorflow.keras import Model
import numpy as np
from cleverhans.future.tf2.attacks import fast_gradient_method

def generate_attack(model, data, eps, norm):
	return fast_gradient_method(model_fn=model, x=data, eps=eps, norm=norm)
	
def evaluate(model, test):
	return model.evaluate(x=test[0], y=test[1])[1]
	
def evaluate_regularize(model, test):
	return model.evaluate({'feature': test[0], 'label': test[1]})[2]
