import numpy as np

def add_noise(data, factor):
	return data + (0.1 * np.random.normal(size=data.shape))
