"""
[Reference]
    [Classification]
     - Resnet paper: https://arxiv.org/pdf/1512.03385
"""
import tensorflow as tf
from tensorflow.keras import layers

from ..block import BlockRes
from ..common import compose

args = {
    "res18":{
        "input_shape":(224, 224, 3),
        "filters":[[64,64],[128,128],[256,256],[512,512]],
        "kernel_sizes":[3, 3],
        "iters":[2, 2, 2, 2],
    },
    "res34":{
        "input_shape":(224, 224, 3),
        "filters":[[64,64],[128,128],[256,256],[512,512]],
        "kernel_sizes":[3, 3],
        "iters":[3, 4, 6, 3],
    },
    "res50":{
        "input_shape":(224, 224, 3),
        "filters":[[64, 64, 256],[128, 128, 512], [256, 256, 1024], [512, 512, 2048]],
        "kernel_sizes":[1, 3, 1],
        "iters":[3, 4, 6, 3],
    },
    "res102":{
        "input_shape":(224, 224, 3),
        "filters":[[64, 64, 256],[128, 128, 512], [256, 256, 1024], [512, 512, 2048]],
        "kernel_sizes":[1, 3, 1],
        "iters":[3, 4, 23, 3],
    },
    "res152":{
        "input_shape":(224, 224, 3),
        "filters":[[64, 64, 256],[128, 128, 512], [256, 256, 1024], [512, 512, 2048]],
        "kernel_sizes":[1, 3, 1],
        "iters":[3, 8, 36, 3],
    },
}

def get_model(input_shape, filters, iters, kernel_sizes, num_classes, activation):
    inp=layers.Input(shape=input_shape)
    x=compose(
        layers.ZeroPadding2D(padding=3),
        layers.Conv2D(64, 7, strides=2),
        layers.BatchNormalization(epsilon=1.001e-05),
        layers.Activation("relu"),
        layers.ZeroPadding2D( ),
        layers.MaxPooling2D(pool_size=3,strides=2)
    )(inp)
    
    for i, (filter, iter) in enumerate(zip(filters, iters)):
        for j in range(iter):
            x = BlockRes(
                x, filter, kernel_sizes, 
                strides = 2 if (i != 0) &( j==0) else 1)
    x = layers.GlobalAvgPool2D()(x)
    out = layers.Dense( num_classes, activation = "softmax")(x)
    return tf.keras.Model(inp, out)
