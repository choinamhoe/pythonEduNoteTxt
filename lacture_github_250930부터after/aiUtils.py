import numpy as np
import tensorflow as tf

class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self,df, batch_size, fun):
        self.data = df
        self.batch_size = batch_size
        self.fun = fun
    def __len__(self):
        return np.ceil(
            self.data.shape[0]/self.batch_size).astype(int)
    def __getitem__(self,index):
        st = index * self.batch_size
        ed = (index+1)* self.batch_size
        values = self.data.values[st:ed]
        x_list = []
        y_list = []
        for value in values:
            x, y = self.fun(value[0])
            x_list.append(x)
            y_list.append(y)
        batch_x = np.array(x_list)
        batch_y = np.array(y_list)
        return batch_x, batch_y
    def on_epoch_end(self):
        self.data = self.data.sample(frac = 1)