"""
[Reference]
    [Classification]
     - Vgg paper: https://arxiv.org/pdf/1409.1556
"""
import tensorflow as tf
from tensorflow.keras import layers

from ..block import BlockVgg
from ..common import compose

args = {
    "vgg11":{
        "filters":[64, 128, 256, 512, 512], 
        "iters":[1, 1, 2, 2, 2],
        "body_filters":[4096, 4096]
    },
    "vgg13":{
        "filters":[64, 128, 256, 512, 512], 
        "iters":[2, 2, 2, 2, 2],
        "body_filters":[4096, 4096]
    },
    "vgg16":{
        "filters":[64, 128, 256, 512, 512], 
        "iters":[2, 2, 3, 3, 3],
        "body_filters":[4096, 4096]
    },
    "vgg19":{
        "filters":[64, 128, 256, 512, 512], 
        "iters":[2, 2, 4, 4, 4],
        "body_filters":[4096, 4096]
    },
}

def get_model(
    input_shape, filters, iters, body_filters, num_classes =1000,
    flat_fun = layers.Flatten, activation = "softmax", kernel_size = 3):

    header_blocks = [BlockVgg(
        filter, [kernel_size]*iter
    ) for filter, iter in zip(filters, iters)]
    
    acts = ["relu"] * (len(filters) - 1) + [activation]
    dense_layers = [
        layers.Dense(i, activation=j)
        for i,j in zip(body_filters, acts)]
    dense_layers.append(
        layers.Dense(num_classes, activation))
    blocks = [flat_fun()] + dense_layers
    body_blocks = compose(*blocks)

    inp = layers.Input(shape = input_shape)
    x = compose(*header_blocks)(inp)
    out = body_blocks(x)    
    return tf.keras.Model(inp, out)
