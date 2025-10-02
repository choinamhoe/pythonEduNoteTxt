import tensorflow as tf

dir(tf.keras.applications)
model = tf.keras.applications.VGG16()
model.summary()
model.layers[:4]

dir(model.layers[1])
model.layers[1].kernel_size
model.layers[1].filters

"""
inp = tf.keras.layers.Input(shape=(224,224,3))
x = tf.keras.layers.Conv2D(64,kernel_size=3, activation="ReLU")(inp)
x = tf.keras.layers.Conv2D(64,kernel_size=3, activation="ReLU")(x)
x = tf.keras.layers.MaxPool2D(pool_size=2)(x)  # pool_size=2 기본값
"""
# 12 시 10분까지
# batch norm 이 없는 이유는 아직 안나와서
inp = tf.keras.layers.Input(shape=(224,224,3))
# block1
x = tf.keras.layers.Conv2D(64,kernel_size=3, activation="ReLU",padding="same")(inp)
x = tf.keras.layers.Conv2D(64,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

# block 2
x = tf.keras.layers.Conv2D(128,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.Conv2D(128,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

# block 3
x = tf.keras.layers.Conv2D(256,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.Conv2D(256,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.Conv2D(256,kernel_size=1, activation="ReLU",padding="same")(x)
x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

# block 4
x = tf.keras.layers.Conv2D(512,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.Conv2D(512,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.Conv2D(512,kernel_size=1, activation="ReLU",padding="same")(x)
x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
# block 5
x = tf.keras.layers.Conv2D(512,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.Conv2D(512,kernel_size=3, activation="ReLU",padding="same")(x)
x = tf.keras.layers.Conv2D(512,kernel_size=1, activation="ReLU",padding="same")(x)
x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

x = tf.keras.layers.Flatten()(x)
out = tf.keras.layers.Dense(1000, activation="softmax")(x)
model = tf.keras.Model(inp,out)
model.summary()
"""
def fun(inp, filters, kernels):
    
    x=layer1(inp)
    x=layer2(x)
    ...
    out = layern(x)
    return out
"""

def vgg_block(inp, filter_size, kernel_list):
    x = inp
    for kernel in kernel_list:
        x = tf.keras.layers.Conv2D(
            filter_size, kernel_size=kernel, 
            activation="ReLU",padding="same")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    return x

#FLatten 썼을 때 파라미터 수 3500만
inp = tf.keras.layers.Input(shape=(224,224,3))
x = vgg_block(inp, 64, [3,3])
x = vgg_block(x, 128, [3,3])
x = vgg_block(x, 256, [3,3,1])
x = vgg_block(x, 512, [3,3,1])
x = vgg_block(x, 512, [3,3,1])
x = tf.keras.layers.Flatten()(x)
out = tf.keras.layers.Dense(1000, activation="softmax")(x)
model = tf.keras.Model(inp,out)
tf.keras.Model(inp, out).summary()


# GAP 썼을 때 파라미터 수 1000만
inp = tf.keras.layers.Input(shape=(224,224,3))
x = vgg_block(inp, 64, [3,3])
x = vgg_block(x, 128, [3,3])
x = vgg_block(x, 256, [3,3,1])
x = vgg_block(x, 512, [3,3,1])
x = vgg_block(x, 512, [3,3,1])
x = tf.keras.layers.GlobalAveragePooling2D()(x)
out = tf.keras.layers.Dense(1000, activation="softmax")(x)
model = tf.keras.Model(inp,out)
tf.keras.Model(inp, out).summary()


### Google net(Inception Net) 블록 간단 예시
inp = tf.keras.layers.Input(shape=(224,224,3))
x1 = tf.keras.layers.Conv2D(
    64,kernel_size=1, activation="ReLU",padding="same")(inp)
x2 = tf.keras.layers.Conv2D(
    64,kernel_size=3, activation="ReLU",padding="same")(inp)
x3 = tf.keras.layers.Conv2D(
    64,kernel_size=5, activation="ReLU",padding="same")(inp)
out = tf.keras.layers.Concatenate()([x1,x2,x3])
tf.keras.Model(inp,out).summary()


### 2시 10분 시작
# 
from functools import reduce
def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

inp = tf.keras.layers.Input(shape=(224,224,3))
conv1 = tf.keras.layers.Conv2D(
    64,kernel_size=1, activation="ReLU",padding="same")
conv2 = tf.keras.layers.Conv2D(
    64,kernel_size=1, activation="ReLU",padding="same")

out = conv1(inp)
conv2(out)

out = compose(*[conv1,conv2])(inp)
tf.keras.Model(inp,out).summary()

# before
def vgg_block_v0(inp, filter_size, kernel_list):
    x = inp
    for kernel in kernel_list:
        x = tf.keras.layers.Conv2D(
            filter_size, kernel_size=kernel, 
            activation="ReLU",padding="same")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    return x
inp = tf.keras.layers.Input(shape=(224,224,3))
x = vgg_block_v0(inp, 64, [3,3])
tf.keras.Model(inp, x).summary()


def vgg_block_v1(filter_size, kernel_list):
    layer_list = []
    for kernel in kernel_list:
        layer = tf.keras.layers.Conv2D(
            filter_size, kernel_size=kernel, 
            activation="ReLU",padding="same")
        layer_list.append(layer)
    layer = tf.keras.layers.MaxPool2D(pool_size=2)
    layer_list.append(layer)
    return compose(*layer_list)

inp = tf.keras.layers.Input(shape=(224,224,3))
x = vgg_block_v1(64, [3,3])(inp)
tf.keras.Model(inp, x).summary()

## 23분 시작