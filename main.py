import numpy as np

import load_data
import load_model
import train
import attack
import noise

# Hyperparameters
epochs = 20
adv_mult = 0.2
adv_step = 0.05
attack_eps = 0.03
attack_norm = np.inf
noise_factor = 0.01

(x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_data.load_fashion_mnist()
x_train_noise_001 = noise.add_noise(x_train, 0.01)
x_train_noise_005 = noise.add_noise(x_train, 0.05)
x_train_noise_01 = noise.add_noise(x_train, 0.1)
x_test_noise_001 = noise.add_noise(x_test, 0.01)
x_test_noise_005 = noise.add_noise(x_test, 0.05)
x_test_noise_01 = noise.add_noise(x_test, 0.1)

data_models = []

for selected_train_data in [x_train, x_train_noise_001, x_train_noise_005, x_train_noise_01]:
	for selected_model in [load_model.lenet_base, load_model.lenet_max, load_model.lenet_deep5, load_model.lenet_max_deep5, load_model.lenet_large, load_model.lenet_max_large, load_model.lenet_deep5_large, load_model.lenet_max_deep5_large, load_model.lenet_filters2, load_model.lenet_max_filters2]:

		model = train.train(selected_model(), epochs, (selected_train_data, y_train_cat), (x_test, y_test_cat))
		adv_model = train.train_regularize(selected_model(), epochs, adv_mult, adv_step, (selected_train_data, y_train), (x_test, y_test))

		x_test_adv = attack.generate_attack(model, x_test, attack_eps, attack_norm)
		x_test_noise_001_adv = noise.add_noise(x_test_adv, 0.01)
		x_test_noise_005_adv = noise.add_noise(x_test_adv, 0.05)
		x_test_noise_01_adv = noise.add_noise(x_test_adv, 0.1)

		data_model_eval = []

#		data_model_eval.append(attack.evaluate(model, (selected_train_data, y_train_cat)))
		data_model_eval.append(attack.evaluate(model, (x_test, y_test_cat)))
#		data_model_eval.append(attack.evaluate_regularize(adv_model, (selected_train_data, y_train)))
		data_model_eval.append(attack.evaluate_regularize(adv_model, (x_test, y_test)))
		data_model_eval.append(attack.evaluate(model, (x_test_noise_001, y_test_cat)))
		data_model_eval.append(attack.evaluate(model, (x_test_noise_005, y_test_cat)))
		data_model_eval.append(attack.evaluate(model, (x_test_noise_01, y_test_cat)))
		data_model_eval.append(attack.evaluate_regularize(adv_model, (x_test_noise_001, y_test)))
		data_model_eval.append(attack.evaluate_regularize(adv_model, (x_test_noise_005, y_test)))
		data_model_eval.append(attack.evaluate_regularize(adv_model, (x_test_noise_01, y_test)))
		data_model_eval.append(attack.evaluate(model, (x_test_adv, y_test_cat)))
		data_model_eval.append(attack.evaluate_regularize(adv_model, (x_test_adv, y_test)))
		data_model_eval.append(attack.evaluate(model, (x_test_noise_001_adv, y_test_cat)))
		data_model_eval.append(attack.evaluate(model, (x_test_noise_005_adv, y_test_cat)))
		data_model_eval.append(attack.evaluate(model, (x_test_noise_01_adv, y_test_cat)))
		data_model_eval.append(attack.evaluate_regularize(adv_model, (x_test_noise_001_adv, y_test)))
		data_model_eval.append(attack.evaluate_regularize(adv_model, (x_test_noise_005_adv, y_test)))
		data_model_eval.append(attack.evaluate_regularize(adv_model, (x_test_noise_01_adv, y_test)))

		data_models.append(data_model_eval)
		
		print("#################")
		for data_model_el in data_model_eval[:-1]:
			print("%.4f" % data_model_el, end=",")
		print("%.4f" % data_model_eval[-1])
		print("#################")

print("#################")
print("# Result Matrix #")
print("#################")
for data_model in data_models:
	for data_model_el in data_model[:-1]:
		print("%.4f" % data_model_el, end=",")
	print("%.4f" % data_model[-1])
print("#################")
