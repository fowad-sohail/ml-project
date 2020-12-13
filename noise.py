import numpy as np

def add_noise(data, factor):
	return data + (factor * np.random.normal(size=data.shape))
