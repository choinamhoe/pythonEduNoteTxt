"""
[Reference]
    [Classification]
     - Efficientnet paper: https://arxiv.org/pdf/1905.11946
"""
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations

from ..layers import EfficientConv2D, EfficientDepthwiseConv2D, EfficientDense
from ..block import CBNEfficient, MBConvBlock, SEBlock
from ..common import compose

args = {
    "base":{
        "kernel_sizes":[3, 3, 5, 3, 5, 5, 3],
        "iters":[1, 2, 2, 3, 3, 4, 1],
        "strides":[1, 2, 2, 2, 1, 2, 1],
        "output_filters": [16, 24, 40, 80, 112, 192, 320],
        "depth_divisor":8,
        "se_ratio":4
    },
    "added":{
        "B0":{
            "input_shape":(224,224,3),
            "width_coefficient":1.,
            "depth_coefficient":1.,
            "drop_connect_rate":0.2,
        },
        "B1":{
            "input_shape":(240,240,3),
            "width_coefficient":1.,
            "depth_coefficient":1.1,
            "drop_connect_rate":0.2,
        },
        "B2":{
            "input_shape":(260,260,3),
            "width_coefficient":1.1,
            "depth_coefficient":1.2,
            "drop_connect_rate":0.3,
        },
        "B3":{
            "input_shape":(300,300,3),
            "width_coefficient":1.2,
            "depth_coefficient":1.4,
            "drop_connect_rate":0.3,
        },
        "B4":{
            "input_shape":(380,380,3),
            "width_coefficient":1.4,
            "depth_coefficient":1.8,
            "drop_connect_rate":0.4,
        },
        "B5":{
            "input_shape":(456,456,3),
            "width_coefficient":1.6,
            "depth_coefficient":2.2,
            "drop_connect_rate":0.4,
        },
        "B6":{
            "input_shape":(528,528,3),
            "width_coefficient":1.8,
            "depth_coefficient":2.6,
            "drop_connect_rate":0.5,
        },
        "B7":{
            "input_shape":(600,600,3),
            "width_coefficient":2.,
            "depth_coefficient":3.1,
            "drop_connect_rate":0.5,
        },
    }
}
def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    return int(np.ceil(depth_coefficient * repeats))

def get_model(
    input_shape, kernel_sizes, iters, strides, output_filters, num_classes,
    width_coefficient = 1., depth_coefficient = 1.,
    drop_connect_rate = 0.2, depth_divisor = 8, se_ratio = 4,
    activation = "softmax"
):
    inp = layers.Input(shape = input_shape)
    filter = round_filters(32, width_coefficient, depth_divisor)
    x = layers.Rescaling(scale=0.00392156862745098)(inp)
    x = layers.Normalization(axis=-1)(x)
    x = CBNEfficient(filter, name = "stem", padding = "same", strides = 2)(x)
    
    block_num = 0
    num_blocks_total = sum(iters)
    alphabet = string.ascii_lowercase
    for idx, (kernel_size, iter, stride) in enumerate(zip(kernel_sizes, iters, strides)):
        name = f"block{idx+1}a"
        if idx==0:
            input_filter = output_filters[idx]*2
        else:
            input_filter = output_filters[idx-1]
        output_filter = output_filters[idx]
        
        num_repeat = round_repeats(iter, depth_coefficient)
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        input_filters = round_filters(input_filter, width_coefficient, depth_divisor)
        expand_ratio = 1 if idx == 0 else 6
        channels = x.shape[-1]
        expand_channels = channels * expand_ratio
        
        x = MBConvBlock(
            x, expand_channels, name = name, 
            first_block = expand_ratio==1, kernel_size=kernel_size,
            strides = stride
        )
        x = SEBlock(x, expand_channels, max(1, channels//se_ratio), name) 
        x = EfficientConv2D(output_filter, 1, name=f"{name}_project_conv")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_project_bn")(x)
        
        block_num += 1
        if num_repeat > 1:
            input_filter = output_filter
            for bidx in range(num_repeat - 1):
                block_name = alphabet[1:][bidx]
                name=f"block{idx+1}{block_name}"
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
    
                channels = x.shape[-1]
                expand_channels = channels * expand_ratio
                # Residual 
                _x = x
                x = MBConvBlock(
                    x, expand_channels, name = name, 
                    first_block = expand_ratio==1, kernel_size=kernel_size,
                )
                x = SEBlock(x, expand_channels, max(1, channels//se_ratio), name) 
                x = EfficientConv2D(output_filter, 1, name=f"{name}_project_conv")(x)
                x = tf.keras.layers.BatchNormalization(name=f"{name}_project_bn")(x)
                x = layers.Dropout(
                    rate=drop_rate, noise_shape=(
                        None, 1, 1, 1),name=f"{name}_drop")(x)
                x = layers.Add(name=f"{name}_add")([_x, x])
                
    x=EfficientConv2D(1280,1,name="top_conv")(x)
    x=tf.keras.layers.BatchNormalization(name="top_bn")(x)
    x=tf.keras.layers.Activation(
        tf.keras.activations.swish,name="top_activation")(x)
    x=layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x=tf.keras.layers.Dropout(
        rate= 0.2,name="top_dropout")(x)
    out=EfficientDense(num_classes, activation= activation,name="predictions")(x)
    return tf.keras.Model(inp, out)

# args = base_dict.copy()
# args.update(added_dict["B0"])
# B0 = get_model(**args)

# args = base_dict.copy()
# args.update(added_dict["B1"])
# B1 = get_model(**args)

# args = base_dict.copy()
# args.update(added_dict["B2"])
# B2 = get_model(**args)

# args = base_dict.copy()
# args.update(added_dict["B3"])
# B3 = get_model(**args)

# args = base_dict.copy()
# args.update(added_dict["B4"])
# B4 = get_model(**args)

# args = base_dict.copy()
# args.update(added_dict["B5"])
# B5 = get_model(**args)

# args = base_dict.copy()
# args.update(added_dict["B6"])
# B6 = get_model(**args)

# args = base_dict.copy()
# args.update(added_dict["B7"])
# B7 = get_model(**args)

