import sys
sys.path.append("E:/cjcho_work/250930/knowledge_distillation")

from utils.classification.basic import ClassificationBuilder
from utils.classification import vgg, res, mobile, efficient

builder = ClassificationBuilder(
    num_classes = 10, input_shape = (224,224,3),
    activation="softmax"
    )

model = builder.build(vgg.args["vgg19"], vgg.get_model)

args = {
     "filters":[32, 64, 128, 256, 256],
     "iters":[1,1,2,2,2],
     'input_shape': (224, 224, 3),
     'body_filters': [4096, 4096],
     'num_classes': 10,
     'activation': 'softmax'
 }
model = builder.build(args, vgg.get_model)

model.summary()

## resnet 
res.args
model = builder.build(res.args["res18"], res.get_model)
model.summary()
## 4시 10분 
mobile.args
model = builder.build(mobile.args, mobile.get_model)
model.summary()

args = efficient.args["base"]
args.update(efficient.args["added"]["B0"])
model = builder.build(args, efficient.get_model)
model.summary()
