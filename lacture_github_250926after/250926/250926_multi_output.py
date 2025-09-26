# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 15:10:15 2025

@author: human
"""

import cv2 , tqdm, os, glob, string
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt 

%matplotlib auto

os.chdir("E:/cjcho_work/250926")
tr_files = glob.glob("train/*")
train = pd.DataFrame({"path":tr_files })

te_files = glob.glob("test/*")
test = pd.DataFrame({"path":te_files })

file_path = tr_files[0]
file_path = tr_files[0]
y1 = int(file_path.split("_")[-1].split(".")[0])
y2 = file_path.split("_")[-3].lower() # 글자값
alphabat=list(string.ascii_lowercase) # 알파벳 리스트

np.where([i==y2 for i in alphabat])[0]
alphabat.index(y2)

file_path = tr_files[0]
y1 = int(file_path.split("_")[-1].split(".")[0])
y2 = file_path.split("_")[-3].lower() # 글자값
alphabat=list(string.ascii_lowercase) # 알파벳 리스트
# type 1
np.where([i==y2 for i in alphabat])[0]
# type 2
y2 = alphabat.index(y2)
y1_output = np.zeros(10)
y1_output[y1] = 1

y2_output = np.zeros(26)
y2_output[y2] = 1

def preprocessing(file_path):
    x = cv2.imread(file_path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    
    y1 = int(file_path.split("_")[-1].split(".")[0])
    y2 = file_path.split("_")[-3].lower() # 글자값
    alphabat=list(string.ascii_lowercase) # 알파벳 리스트
    # type 1
    np.where([i==y2 for i in alphabat])[0]
    # type 2
    y2 = alphabat.index(y2)
    y1_output = np.zeros(10)
    y1_output[y1] = 1

    y2_output = np.zeros(26)
    y2_output[y2] = 1
    return x, y1_output, y2_output

x,y1,y2 = preprocessing(file_path)
x.shape, y1.shape, y2.shape
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, preprocessing, aug_fun=None):
        self.data = df
        self.batch_size = batch_size 
        self.preprocessing = preprocessing
        self.aug_fun = aug_fun
    def __len__(self):
        return self.data.shape[0]//self.batch_size
    def __getitem__(self, index):
        st = index * self.batch_size
        ed = (index + 1) * self.batch_size
        paths = self.data["path"].values[st:ed]
        x_list = []
        y1_list = []
        y2_list = []
        for file_path in paths:
            x, y1, y2 = self.preprocessing(file_path)
            if self.aug_fun:
                x = np.array([aug_fun(image=i)["image"] for i in x])
            x_list.append(x)
            y1_list.append(y1)
            y2_list.append(y2)
        bat_x = np.array(x_list)
        bat_y1 = np.array(y1_list)
        bat_y2 = np.array(y2_list)
        return bat_x, (bat_y1, bat_y2)
def on_epoch_end(self):
        self.data = self.data.sample(frac = 1)