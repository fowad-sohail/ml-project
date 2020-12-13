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
attack_eps = 0.3
attack_norm = np.inf
noise_factor = 0.01

(x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_data.load_fashion_mnist()
x_train_noise_001 = noise.add_noise(x_train, 0.01)
x_train_noise_005 = noise.add_noise(x_train, 0.05)
x_train_noise_01 = noise.add_noise(x_train, 0.1)
x_test_noise_001 = noise.add_noise(x_test, 0.01)
x_test_noise_005 = noise.add_noise(x_test, 0.05)
x_test_noise_01 = noise.add_noise(x_test, 0.1)

# Models:
# lenet_base, lenet_max, lenet_deep5, lenet_max_deep5, lenet_large, lenet_max_large, lenet_deep5_large, lenet_max_deep5_large, lenet_filters2, lenet_max_filters2
selected_model = load_model.lenet_base

# Train Data:
# x_train, x_train_noise_001, x_train_noise_005, x_train_noise_01
selected_train_data = x_train

model = train.train(selected_model(), epochs, (selected_train_data, y_train_cat), (x_test, y_test_cat))
adv_model = train.train_regularize(selected_model(), epochs, adv_mult, adv_step, (selected_train_data, y_train), (x_test, y_test))

#print("0.01 Noise in Model", attack.evaluate(model, (x_test_noise_001, y_test_cat)))
#print("0.01 Noise in Reg Model", attack.evaluate_regularize(adv_model, (x_test_noise_001, y_test)))
#print("0.05 Noise in Model", attack.evaluate(model, (x_test_noise_005, y_test_cat)))
#print("0.05 Noise in Reg Model", attack.evaluate_regularize(adv_model, (x_test_noise_005, y_test)))
#print("0.1 Noise in Model", attack.evaluate(model, (x_test_noise_01, y_test_cat)))
#print("0.1 Noise in Reg Model", attack.evaluate_regularize(adv_model, (x_test_noise_01, y_test)))

x_test_adv = attack.generate_attack(model, x_test, attack_eps, attack_norm)
x_test_noise_001_adv = noise.add_noise(x_test_adv, 0.01)
x_test_noise_005_adv = noise.add_noise(x_test_adv, 0.05)
x_test_noise_01_adv = noise.add_noise(x_test_adv, 0.1)

#print("Attacked Model", attack.evaluate(model, (x_test_adv, y_test_cat)))
#print("Attacked Reg Model", attack.evaluate_regularize(adv_model, (x_test_adv, y_test)))
#print("0.01 Noise in Attacked Model", attack.evaluate(model, (x_test_noise_001_adv, y_test_cat)))
#print("0.01 Noise in Attacked Reg Model", attack.evaluate_regularize(adv_model, (x_test_noise_001_adv, y_test)))
#print("0.05 Noise in Attacked Model", attack.evaluate(model, (x_test_noise_005_adv, y_test_cat)))
#print("0.05 Noise in Attacked Reg Model", attack.evaluate_regularize(adv_model, (x_test_noise_005_adv, y_test)))
#print("0.1 Noise in Attacked Model", attack.evaluate(model, (x_test_noise_01_adv, y_test_cat)))
#print("0.1 Noise in Attacked Reg Model", attack.evaluate_regularize(adv_model, (x_test_noise_01_adv, y_test)))

base_train = attack.evaluate(model, (selected_train_data, y_train_cat))
base_test = attack.evaluate(model, (x_test, y_test_cat))
adv_train = attack.evaluate_regularize(adv_model, (selected_train_data, y_train))
adv_test = attack.evaluate_regularize(adv_model, (x_test, y_test))
noise_base_001 = attack.evaluate(model, (x_test_noise_001, y_test_cat))
noise_base_005 = attack.evaluate(model, (x_test_noise_005, y_test_cat))
noise_base_01 = attack.evaluate(model, (x_test_noise_01, y_test_cat))
noise_adv_001 = attack.evaluate_regularize(adv_model, (x_test_noise_001, y_test))
noise_adv_005 = attack.evaluate_regularize(adv_model, (x_test_noise_005, y_test))
noise_adv_01 = attack.evaluate_regularize(adv_model, (x_test_noise_01, y_test))
base_attack = attack.evaluate(model, (x_test_adv, y_test_cat))
adv_attack = attack.evaluate_regularize(adv_model, (x_test_adv, y_test))
noise_base_001_attack = attack.evaluate(model, (x_test_noise_001_adv, y_test_cat))
noise_base_005_attack = attack.evaluate(model, (x_test_noise_005_adv, y_test_cat))
noise_base_01_attack = attack.evaluate(model, (x_test_noise_01_adv, y_test_cat))
noise_adv_001_attack = attack.evaluate_regularize(adv_model, (x_test_noise_001_adv, y_test))
noise_adv_005_attack = attack.evaluate_regularize(adv_model, (x_test_noise_005_adv, y_test))
noise_adv_01_attack = attack.evaluate_regularize(adv_model, (x_test_noise_01_adv, y_test))

print("################")
print("# Data to copy #")
print("################")
print("%.4f" % base_train, end=",")
print("%.4f" % base_test, end=",")
print("%.4f" % adv_train, end=",")
print("%.4f" % adv_test, end=",")
print("%.4f" % noise_base_001, end=",")
print("%.4f" % noise_base_005, end=",")
print("%.4f" % noise_base_01, end=",")
print("%.4f" % noise_adv_001, end=",")
print("%.4f" % noise_adv_005, end=",")
print("%.4f" % noise_adv_01, end=",")
print("%.4f" % base_attack, end=",")
print("%.4f" % adv_attack, end=",")
print("%.4f" % noise_base_001_attack, end=",")
print("%.4f" % noise_base_005_attack, end=",")
print("%.4f" % noise_base_01_attack, end=",")
print("%.4f" % noise_adv_001_attack, end=",")
print("%.4f" % noise_adv_005_attack, end=",")
print("%.4f" % noise_adv_01_attack)
print("################")
