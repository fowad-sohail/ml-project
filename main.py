import tensorflow as tf


model = tf.keras.applications.EfficientNetB7(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000, classifier_activation='softmax'
)

data_dir = tf.keras.utils.get_file(origin='/data/pytorch/',
                                   fname='cifar-10-python.tar.gz', 
                                   untar=True)

print(model)
print(data_dir)