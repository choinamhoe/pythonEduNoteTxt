import tensorflow as tf
from tensorflow.keras import layers
from functools import wraps

conv_regularizer=tf.keras.initializers.VarianceScaling(
    scale=2.0,
    mode='fan_out',
    distribution='normal',
    seed=None
)

dense_regularizer=tf.keras.initializers.VarianceScaling(
    scale=1/3,
    mode='fan_out',
    distribution='uniform',
    seed=None
)

@wraps(layers.Conv2D)
def EfficientConv2D(*args, **kwargs):
    """Wrapper to set parameters for Convolution2D."""
    conv_kwargs = {
        'kernel_initializer':conv_regularizer}
    conv_kwargs['padding'] = "same"
    conv_kwargs['use_bias'] = False
    conv_kwargs.update(kwargs)
    return layers.Conv2D(*args, **conv_kwargs)

@wraps(layers.DepthwiseConv2D)
def EfficientDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set parameters for Convolution2D."""
    conv_kwargs = {
        'depthwise_initializer':conv_regularizer}
    conv_kwargs['padding'] = "same"
    conv_kwargs['use_bias'] = False
    conv_kwargs.update(kwargs)
    return layers.DepthwiseConv2D(*args, **conv_kwargs)

@wraps(layers.Dense)
def EfficientDense(*args, **kwargs):
    """Wrapper to set parameters for Dense."""
    conv_kwargs = {
        'kernel_initializer':dense_regularizer}
    conv_kwargs.update(kwargs)
    return layers.Dense(*args, **conv_kwargs)