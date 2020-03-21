# Created by:
# Name : Alan Fhajoeng Ramadhan
# From : Indonesia
# email : alfhatech.id@gmail.com
# 21 March 2020
# usage : python train_data.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X / 255.0

dense_layers = [0]
layer_sizes = [32]
conv_layers = [2]

for dense_layer in dense_layers:
   for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir= r'Your path\logs\{}'.format(NAME)) # make your logs directory path 
            print(NAME)
            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

            model.add(Flatten())
            for j in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dropout(0.2))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

            model.fit(X, y, batch_size=32, epochs=20, validation_split=0.2, callbacks=[tensorboard])
            
model.save('32x2x0-CNN.model')

Created by:
Name : Alan Fhajoeng Ramadhan
From : Indonesia
email : alfhatech.id@gmail.com
21 March 2020
usage : python train_data.py





