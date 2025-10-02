# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:49:12 2025

@author: Changje Cho

[Reference]
    [Classification]
     - Vgg paper: https://arxiv.org/pdf/1409.1556
     - Inception paper: https://arxiv.org/pdf/1409.4842
     - Resnet paper: https://arxiv.org/pdf/1512.03385
     - Densenet paper: https://arxiv.org/pdf/1608.06993
     - Mobilenet paper: https://arxiv.org/pdf/1704.04861
     - Squeezenet paper: https://arxiv.org/pdf/1602.07360
     - Efficientnet paper: https://arxiv.org/pdf/1905.11946
    
    [Segmentation]
     - UNET paper: https://arxiv.org/pdf/1505.04597
     - FPN: http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
     - LINKNET paper: https://arxiv.org/pdf/1707.03718
     - PSPNET paper: https://arxiv.org/pdf/1612.01105
     - REF: https://github.com/qubvel-org/segmentation_models.pytorch

    [ObjectDetection]
    - RCNN paper: https://arxiv.org/pdf/1311.2524
    - Fastter RCNN paper: https://arxiv.org/pdf/1506.01497
    - YOLO v1 paper: https://arxiv.org/pdf/1506.02640
    - YOLO v3 paper: https://arxiv.org/pdf/1804.02767
    - SSD paper: https://arxiv.org/pdf/1512.02325
    - RetinaNet paper: https://arxiv.org/pdf/1708.02002
    - Mask RCNN paper: https://arxiv.org/pdf/1703.06870

    [FacialLandmarkDetection]
    - BlazeFace paper: https://arxiv.org/pdf/1907.05047
    - RetinaFace paper: https://arxiv.org/pdf/1905.00641
    - Insightface git: https://github.com/deepinsight/insightface?tab=readme-ov-file
    
    [Generation]
    - Unconditional GAN
        - WGAN paper: https://arxiv.org/pdf/1701.07875
        - LSGAN paper: https://arxiv.org/pdf/1611.04076
        - BEGAN paper: https://arxiv.org/pdf/1703.10717
        - SNGAN paper: https://arxiv.org/pdf/1802.05957
        - SAGAN paper: https://arxiv.org/pdf/1805.08318
        - BIGGAN paper: https://arxiv.org/pdf/1809.11096
        - ProgressiveGAN paper: https://arxiv.org/pdf/1710.10196
        - StyleGAN paper: https://arxiv.org/abs/1812.04948
    - Conditional GAN
        - PIX2PIX paper: https://arxiv.org/pdf/1611.07004
        - CycleGAN paper: https://arxiv.org/pdf/1703.10593
        - DiscoGAN paper: https://arxiv.org/pdf/1703.05192
        - GauGAN paper: https://arxiv.org/pdf/1903.07291
    - SRGAN paper: https://arxiv.org/pdf/1609.04802

    [Transfer]
    - Knowledge Distillation paper: https://arxiv.org/pdf/1503.02531
"""
import tensorflow as tf
from tensorflow.keras import layers, activations, Sequential

from .activation import Relu
from .common import compose
from .layers import EfficientConv2D, EfficientDepthwiseConv2D, EfficientDense

def CBN(
    filters, kernel_size=3,
    activation='relu', strides=1,
    padding="valid"):
    
    conv=layers.Conv2D(
        filters, kernel_size = kernel_size,
        strides = strides, padding = padding)
    bn=layers.BatchNormalization(epsilon = 1.001e-05)
    layers_list = [conv, bn]
    if activation:
        act=layers.Activation(activation = activation)
        layers_list.append(act)
    return compose(*layers_list)

def CBNEfficient(
    filters, name, kernel_size=3, strides=1, padding="same"):
    return compose(
        EfficientConv2D(
            filters, kernel_size=kernel_size,
            strides=strides, padding=padding, name=f"{name}_conv"),
        layers.BatchNormalization(name=f"{name}_bn"),
        layers.Activation(
            activations.swish,name=f"{name}_activation"))

def BlockVgg(
    filters, kernel_sizes
    ):
    if not isinstance(filters, list):
        # filters 가 list 가 아니면 broadcast
        filters = [filters] * len(kernel_sizes)

    # kernel_size 개수 만큼 Conv2D 레이어 생성
    conv_layers = [
        layers.Conv2D(
            i, kernel_size=j, padding='same',
            activation='relu')
        for i, j in zip(filters, kernel_sizes)]
    layers_list = conv_layers + [layers.MaxPooling2D()]
    return compose(*layers_list)


def BlockRes(
    x, filters, kernel_sizes, strides = 1
    ):
    blocks = list()
    for idx, (filter, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        stride = strides if idx == 0 else 1
        pad = "same" if idx != len(
            kernel_sizes) or len(kernel_sizes) == 2 else "valid"
        act = None if idx == len(kernel_sizes) else "relu"
        blocks.append(
            CBN(
                filter, kernel_size=kernel_size, 
                padding=pad, activation = act, strides = stride))            
    x1 = compose(*blocks)(x)
    if strides!=1 or (x.shape[-1]!=filters[-1]):
        x = CBN(
            filter, kernel_size=kernel_size, 
            strides = strides, activation = act, padding = pad)(x)
    x = layers.Add()([x1, x])
    x = layers.Activation(activations.relu)(x)
    return x

def BlockMobile(
    kernel_size, filters, 
    padding="valid", 
    strides=1, activation=Relu
    ):

    layers_list=[
        layers.DepthwiseConv2D(
            kernel_size, strides=strides, 
            padding = padding, use_bias=False),
        layers.BatchNormalization( ),
        layers.Activation(activation=Relu),
        # 인셉션 넷과 동일하게 마지막에 1x1 필터를 적용
        layers.Conv2D(filters, 1, padding="same", use_bias=False), 
        layers.BatchNormalization( ),
        layers.Activation(activation=Relu),
    ]
    block=Sequential(layers_list)
    return block

def SEBlock(x, filters, reduced_filters, name):
        """
        Squeeze and Excitation Block
        """
        blocks = compose(
            layers.GlobalAveragePooling2D(
                name = f"{name}_se_squeeze"),
            layers.Reshape((1, 1, -1), name = f"{name}_se_reshape"),
            EfficientConv2D(
                reduced_filters, 1, strides=1, name=f"{name}_se_reduce", 
                use_bias=True, activation = tf.keras.activations.swish),
            EfficientConv2D(
                filters, 1, strides=1, name=f"{name}_se_expand", 
                use_bias=True, activation = tf.keras.activations.sigmoid),        
        )
        out = blocks(x)
        return layers.Multiply(name = f"{name}_se_excite")([out, x])

def MBConvBlock(
    x, filters, name, first_block,
    kernel_size = 3, strides = 1, padding = "same", padding_size = None):
    """
    MobileNet v3 block
    """
    if not first_block:
        x = CBNEfficient(filters, kernel_size = 1, name = f"{name}_expand")(x)
    if padding_size:
        x=tf.keras.layers.ZeroPadding2D(
            name=f"{name}_dwconv_pad",
            padding=(padding_size, padding_size))(x)
        
    x=EfficientDepthwiseConv2D(
        kernel_size, strides=strides, padding=padding, name=f"{name}_dwconv")(x)
    x=tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x=tf.keras.layers.Activation(
        tf.keras.activations.swish,name=f"{name}_activation")(x)
    return x

