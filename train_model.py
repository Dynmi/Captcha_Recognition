import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from src.model import captcha_loss
import json 
import pickle 

with open("config.json","rb") as f:
  config = json.load(f)
max_chars = config["max_chars"]
char_set = config["char_set"]


loss_collect= []
valid_acc_collect = []
train_acc_collect = []
VALID_X = pickle.load(open("./valid_data/valid_x.pkl","rb"))
VALID_Y = pickle.load(open("./valid_data/valid_y.pkl","rb"))


def get_acc(pred,y):
    max_idx_p = tf.argmax(tf.reshape(pred, [-1, max_chars, len(char_set)]), 2)  
    max_idx_l = tf.argmax(tf.reshape(y, [-1, max_chars, len(char_set)]), 2)  

    correct_pred = tf.equal(max_idx_p, max_idx_l)
    return tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))


def train_model(model, n_steps, CYCLE = 5000, BATCH_SIZE=128, lr=0.00025,show_plot=1):
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),loss=captcha_loss,metrics=[get_acc])
  global S_ID
  S_ID = 0

  stp=0
  while stp!=CYCLE:
      X,Y = get_batch(BATCH_SIZE)
      cur_loss,train_acc = model.train_on_batch(X,Y)
      if stp%256 ==0:
        predY = model.predict(np.array(VALID_X))
        valid_acc = get_acc(predY,VALID_Y)
        print("======>>>at step {},loss is {},train_acc is {:.2f}%, validation accuracy is {:.2f}%".format(stp, cur_loss, train_acc*100, valid_acc*100))
        loss_collect.append(cur_loss)
        valid_acc_collect.append( valid_acc.numpy() )
        train_acc_collect.append( train_acc )
      stp+=1

  # show the training process
  if show_plot: 
    plt.figure(figsize=(14,9))
    plt.plot(range(0,len(valid_acc_collect)),valid_acc_collect,color="green")
    plt.plot(range(0,len(train_acc_collect)),train_acc_collect,color="red")
    # plt.title("OPTIMIZER=ADAM  BATCH_SIZE=128  L_R=0.00025  CYCLES=25000  LS=0.05 ")
    plt.yticks(np.arange(0, 1.03, step=0.04))
    plt.xticks(np.arange(0,len(train_acc_collect),step=len(train_acc_collect)/10))
    plt.grid()
    plt.show()