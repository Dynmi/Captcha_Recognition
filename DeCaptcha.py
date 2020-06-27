'''
    author@DynmiWang
'''

import tensorflow as tf 
from PIL import Image
import numpy as np 
import json,os
from src import DataUtils 

with open("./config.json", "r") as f:
    config = json.load(f)

width = config["img_width"]
height = config["img_height"]

model = tf.keras.models.load_model('./saved_model/capt_net90.h5',compile=False)

def dec_img(img_path):
    img = Image.open(img_path)
    img = img.resize((width, height),Image.ANTIALIAS)
    img_array = np.array(img)  
    x = DataUtils.convert2gray(img_array) /255
    X = [x]
    Y = model.predict(np.array(X))
    return DataUtils.vec2text(Y[0])

def dec_batch(img_src):
    X = []
    for i in os.listdir(img_src):
        img = Image.open(os.path.join(img_src,i))
        img = img.resize((width, height),Image.ANTIALIAS)
        img_array = np.array(img)  
        x = DataUtils.convert2gray(img_array) /255
        X.append(x)
    Y = model.predict(np.array(X))
    Y = [DataUtils.vec2text(y) for y in Y]
    return Y