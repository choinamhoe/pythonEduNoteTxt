"""
[Reference]
    [Classification]
     - Mobilenet paper: https://arxiv.org/pdf/1704.04861
"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential, activations

from ..block import BlockMobile
from ..common import compose

args = {
    "input_shape":(224,224,3),
    "filters":[64, 128, 256, 512, 1024],
    "iters":[1, 2, 2, 6, 2]
}

def get_model(
    input_shape, filters, iters, num_classes,
    flat_fun = layers.GlobalAvgPool2D, activation="softmax"
):
    inp=layers.Input(shape=input_shape)
    x=Sequential(
        [
            layers.Conv2D(32, 3, strides=2, padding="same", use_bias= False),
            layers.BatchNormalization( ),
            layers.Activation(activations.relu),# tf.nn.relu6
        ])(inp)
    
    for i, (filter, iter) in enumerate(zip(filters, iters)):
        for j in range(iter):
            padding = "valid" if i != 0 and j == 0 else "same"
            strides = 2 if i != 0 and j == 0 else 1
            x=BlockMobile(
                3, filter, padding=padding, strides = strides)(x)
        if len(iters) != i+1:
            x=layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = flat_fun()(x)
    out = layers.Dense( num_classes, activation = activation)(x)
    return tf.keras.Model(inp, out)

# get_model(**args, num_classes)