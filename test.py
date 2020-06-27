import tensorflow as tf 
import DeCaptcha
from src import DataUtils 

model = tf.keras.models.load_model("/home/dynmi/Documents/project/Captcha_Recgnition/saved_model/capt_net89.h5",compile=False)

model.summary()

img_src='./sample/'
DataUtils.gen_imgs(src="./sample/", count=100)
Y = DeCaptcha.dec_batch(img_src)

for y in Y:
  print(y)