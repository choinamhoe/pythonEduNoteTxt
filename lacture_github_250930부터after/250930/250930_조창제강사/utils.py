# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:37:43 2025

@author: human
"""
import tensorflow as tf
import numpy as np 
import cv2
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, is_train=True):
        self.df = df
        self.batch_size = batch_size
        self.is_train = is_train
    def __len__(self):
        return np.ceil(self.df.shape[0]/self.batch_size).astype(int)
    
    def __getitem__(self, index):
        st = index * self.batch_size
        ed = (index+1) * self.batch_size
        values = self.df.values[st:ed]
        x_list = []
        y_list = []
        for value in values:
            if self.is_train:
                x, y = self.preprocessing(value, self.is_train)
                y_list.append(y)
            else:
                x = self.preprocessing(value, self.is_train)
            x_list.append(x)
        bat_x = np.array(x_list)
        if self.is_train:
            bat_y = np.array(y_list)
            return bat_x, bat_y
        return bat_x
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1)
        
    def preprocessing(self, value, is_train):
        file_path = value[0]
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224,224))
        if is_train:
            label = value[1]
            y = int(label=="Cat")
            return img, y
        else:
            return img
        

class Distiller(tf.keras.Model):
    def __init__(
            self, student_model, teacher_model,
            student_loss_fun,
            distillation_loss_function = tf.keras.losses.KLDivergence(),
            temperature = 5.0, alpha = 0.5
            ):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.student_loss_fun = student_loss_fun
        self.distillation_loss_fun = distillation_loss_function
        
    def compile(self, optimizer, metrics=None):
        super().compile()
        self.optimizer = optimizer 
        if metrics is not None:
            self.student_metric = metrics
            
    def train_step(self, data):
        x, y = data
        # teacher는 훈련 안함
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            # 일반 loss (hard label)
            student_loss = self.student_loss_fun(y, student_predictions)
            # distillation loss (soft label) # 다중분류 기준, 이진분류는 시그모이드 회귀는 없음
            ## 현재코드는 다지분류라 가정 
            student_soft = tf.nn.softmax(student_predictions / self.temperature)
            teacher_soft = tf.nn.softmax(teacher_predictions / self.temperature) 
            distill_loss = self.distillation_loss_fun(teacher_soft, student_soft)
            # 최종 loss (alpha 가중치)
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss * (self.temperature ** 2)
        
        grads = tape.gradient(loss, self.student.trainable_variables)
        # 오차 역전파
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        
        self.student_metric.update_state(y, tf.nn.softmax(student_predictions))
        return {"loss": loss, "accuracy": self.student_metric.result()}
    
    def test_step(self, data):
        x, y = data
        student_predictions = self.student(x, training=False)
        student_loss = self.student_loss_fun(y, student_predictions)
        self.student_metric.update_state(y, tf.nn.softmax(student_predictions))
        return {"loss": student_loss, "accuracy": self.student_metric.result()}
