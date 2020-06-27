'''
    author@DynmiWang
'''

import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
import json 

with open("../config.json","rb") as f:
    config = json.load(f)

max_chars = config["max_chars"]
charset = config["charset"]

# custom loss function
def captcha_loss(y,pred):
    y_pred = tf.reshape(pred, [-1, max_chars, len(charset)]) 
    y_true = tf.reshape(y, [-1, max_chars, len(charset)]) 
    loss = tf.keras.losses.binary_crossentropy(y_true,y_pred,label_smoothing=0.1)   #(BATCH_SIZE, max_chars,)
    loss = tf.reduce_sum(loss,axis=1)      #(BATCH_SIZE,)
    loss = tf.reduce_mean(loss)        #()
    return  loss


def get_cnn_model(img_shape, output_size, show_info=1):
    model = keras.Sequential(name='Captcha_Net')
    model.add(keras.layers.Conv2D(24,3,input_shape=img_shape,activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(64,3,strides=(2,2),activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(96,3,strides=(2,2),activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1024,activation='relu'))
    model.add(keras.layers.Dense(output_size,activation='sigmoid'))
    
    if show_info:
        model.summary()

    return model
