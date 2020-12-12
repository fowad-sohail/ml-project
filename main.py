import numpy as np

import load_data
import load_model
import train
import attack

# Hyperparameters
epochs = 20
adv_mult = 0.2
adv_step = 0.05
attack_eps = 0.3
attack_norm = np.inf

(x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_data.load_fashion_mnist()

model = train.train(load_model.lenet_base(), epochs, (x_train, y_train_cat), (x_test, y_test_cat))
adv_model = train.train_regularize(load_model.lenet_base(), epochs, adv_mult, adv_step, (x_train, y_train), (x_test, y_test))

x_test_adv = attack.generate_attack(model, x_test, attack_eps, attack_norm)

print("Attacked Model", attack.evaluate(model, (x_test_adv, y_test_cat)))
print("Attacked Reg Model", attack.evaluate_regularize(adv_model, (x_test_adv, y_test)))
