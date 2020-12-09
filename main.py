import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.expand_dims(x_train,-1)
x_test=np.expand_dims(x_test,-1)
x_train, x_test = x_train / 255.0, x_test / 255.0


'''
tf.keras.layers.Conv2D(6, kernel_size=3, strides=1),
tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
tf.keras.layers.ReLU(),
tf.keras.layers.Conv2D(16, kernel_size=3, strides=1),
tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
tf.keras.layers.ReLU(),
'''

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax'),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])



history = model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test), validation_freq=1)
