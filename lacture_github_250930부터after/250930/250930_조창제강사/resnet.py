import tensorflow as tf

dir(tf.keras.applications)
check_model = tf.keras.applications.ResNet50()
check_model.layers[7:15]
from functools import reduce
def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

inp = tf.keras.layers.Input(shape=(224,224,3))
x = tf.keras.layers.Conv2D(
    64, kernel_size=7, strides = 2, padding="same")(inp)
x1 = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
# filter, kernel_size
### block 1 
x = tf.keras.layers.Conv2D(64, 1,padding="same")(x1)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Conv2D(64, 3,padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Conv2D(256, 1,padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)

# x1이랑 x 랑 shape 안맞을 때 
x1 = tf.keras.layers.Conv2D(256, 1,padding="same")(x1)
x1 = tf.keras.layers.BatchNormalization()(x1)
x1 = tf.keras.layers.Activation("relu")(x1)

x = tf.keras.layers.Add()([x1, x])
x = tf.keras.layers.Activation("relu")(x)

### block 함수

"""
kernel_size 여러개 입력
체널 개수도 여러개 입력
"""

def CBN(
        filters, kernel_size=3, strides=1,
        activation=None, padding="same"):
    
    conv = tf.keras.layers.Conv2D(
        filters, kernel_size = kernel_size, 
        strides= strides, padding=padding
        )
    bn = tf.keras.layers.BatchNormalization(
        epsilon = 1.001e-05
        )
    layer_list = [conv, bn]
    if activation:
        act = tf.keras.layers.Activation(
            activation=activation)
        layer_list.append(act)
    return compose(*layer_list)

## 블록 형태로 사용 
"""
Conv
Batch
Conv
Batch
Activation
"""
inp = tf.keras.layers.Input(shape=(224,224,3))
x= CBN(64)(inp)
out = CBN(64, activation="ReLU")(x)
tf.keras.Model(inp, out).summary()

### 입력
inp = tf.keras.layers.Input(shape=(224,224,3))

strides = 1
kernel_sizes = [1, 3, 1]
filters = [64, 64, 256]
blocks = []

for idx, (fil, kernel_size) in enumerate(zip(filters,kernel_sizes)):
    stride = strides if idx==0 else 1
    # idx가 마지막이 아니거나 kernel_size길이가 2이면 padding은 same
    # 그렇지 않으면 valid 
    pad = "same" if idx != len(
        kernel_sizes) or len(kernel_sizes) == 2 else "valid"
    # 마지막 커널에 대해서는 activation function 없이 출력
    act = None if idx == len(kernel_sizes) else "relu"
    
    blocks.append(
        CBN(
            fil, kernel_size = kernel_size,
            padding=pad, activation=act, strides = stride
            )
        )
# Res50 기준
# 최종 출력은 Conv + BN + ReLU + Conv + BN + ReLU + Conv + BN 형태
x1 = compose(*blocks)(inp)

# 입력하는 체널의 수랑 최종 x1의 체널의 수가 다르면 add 연산이 안되므로
# conv 연산을 거쳐서 add 연산이 되게 맞춤
if stride !=1 or (inp.shape[-1]!= filters[-1]):
    x = CBN(
       fil, kernel_size = kernel_size,
       padding=pad, activation=act, strides = stride
       )(inp)
# 더하기 연산 후 출력
x = tf.keras.layers.Add()([x1, x])
x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
x

#############################################
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
    x = tf.keras.layers.Add()([x1, x])
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    return x

#### 
# resnet 50 
inp = tf.keras.layers.Input(shape=(224,224,3))
x=compose(
    tf.keras.layers.ZeroPadding2D(padding=3),
    tf.keras.layers.Conv2D(64, 7, strides=2),
    tf.keras.layers.BatchNormalization(epsilon=1.001e-05),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.ZeroPadding2D( ),
    tf.keras.layers.MaxPooling2D(pool_size=3,strides=2)
)(inp)

x = BlockRes(x, [64,64,256], [1,3,1])
x = BlockRes(x, [64,64,256], [1,3,1])
x = BlockRes(x, [64,64,256], [1,3,1])

x = BlockRes(x, [128,128,512], [1,3,1])
x = BlockRes(x, [128,128,512], [1,3,1])
x = BlockRes(x, [128,128,512], [1,3,1])
x = BlockRes(x, [128,128,512], [1,3,1])

x = BlockRes(x, [256,256,1024], [1,3,1])
x = BlockRes(x, [256,256,1024], [1,3,1])
x = BlockRes(x, [256,256,1024], [1,3,1])
x = BlockRes(x, [256,256,1024], [1,3,1])
x = BlockRes(x, [256,256,1024], [1,3,1])
x = BlockRes(x, [256,256,1024], [1,3,1])

x = BlockRes(x, [512,512,2048], [1,3,1])
x = BlockRes(x, [512,512,2048], [1,3,1])
x = BlockRes(x, [512,512,2048], [1,3,1])
x = tf.keras.layers.GlobalAvgPool2D()(x)
out = tf.keras.layers.Dense(1000)(x)
tf.keras.Model(inp, out).summary()
check_model.summary()
