import tensorflow as tf
from tensorflow.keras import layers

class ClassificationBuilder:
    def __init__(
        self, num_classes, activation, input_shape = (224, 224, 3)):
        self.input_shape = input_shape 
        self.num_classes = num_classes
        self.activation = activation
        
    def build(self, args, model_fun):
        args.update({
            "input_shape":self.input_shape,
            "num_classes":self.num_classes,
            "activation":self.activation,
        })
        return model_fun(**args)
