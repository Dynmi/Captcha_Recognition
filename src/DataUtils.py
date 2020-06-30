'''
    author@DynmiWang
'''

from tensorflow import keras
from captcha.image import ImageCaptcha
from PIL import Image
import random
import numpy as np
import os
import time
import json

with open("./config.json", "r") as f:
    config = json.load(f)

charset=config["charset"]
max_chars = config["max_chars"]
width = config["img_width"]
height = config["img_height"]
output_size = max_chars*len(charset)

'''
generate image files  using ImageCaptcha 
'''
def gen_imgs(src="./sample/", count=40000, width=100, height=60, image_suffix="png"):
    if not os.path.exists(src):
       os.makedirs(src)
    
    generator = ImageCaptcha(width=width, height=height)

    for i in range(count):
        text = ""
        for _ in range(max_chars):
            text += random.choice(charset)
        timec = str(time.time()).replace(".", "")
        save_path = os.path.join(src, "{}_{}.{}".format(text, timec, image_suffix))
        img = generator.generate_image(text)
        img.save(save_path)

        if i%256==0:
            print("Generate image -->> {}".format(i + 1))

'''
convert RGB image to gray image
'''
def convert2gray(img):
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return np.expand_dims(img,axis=2)

'''
convert text to vector
'''
def text2vec(text):
    text_len = len(text)
    if text_len > max_chars: raise ValueError('验证码最长{}个字符'.format(max_chars))
    vec = np.zeros(output_size)
    for i, ch in enumerate(text):
        idx = i * len(charset) + charset.index(ch)
        vec[idx] = 1
    return vec

'''
convert vector to text
'''
def vec2text(vec):
    text=""
    s=0
    while s < max_chars*len(charset):
        char = charset[np.argmax(vec[s:s+len(charset)])]
        text+=str(char)
        s+=len(charset)
    return text

'''
get images and labels from source files
'''
def get_captcha_text_image(src,img_name,zoom=0):
    label = img_name.split("_")[0]
    img= Image.open(src+img_name)
    if zoom: img = img.resize((width, height),Image.ANTIALIAS)
    img_array = np.array(img)  
    img_array = convert2gray(img_array) /255
    return text2vec(label), img_array 

'''
get batch data for training
'''
def get_batch(src, BATCH_SIZE=64):
    X,Y = [],[]
    img_list = os.listdir(src)
    img_list = random.sample(img_list,BATCH_SIZE)
    for i in img_list:
        label,x = get_captcha_text_image(src,i)
        X.append(x)
        Y.append(label)
    return np.array(X),np.array(Y)

