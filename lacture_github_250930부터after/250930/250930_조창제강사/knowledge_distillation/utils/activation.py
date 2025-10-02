import tensorflow.keras.backend as K

def Relu(x):
    return K.relu(x, max_value=6.0, alpha=0, threshold=0)
