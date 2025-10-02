import glob,cv2
import tensorflow as tf

### 중간 레이어에서 출력이 어떻게 나오는지 확인

model = tf.keras.models.load_model(
    "E:/cjcho_work/250930/teacher.h5")

# cat_and_dogs 데이터 있으신분들
files = glob.glob(
    "E:/cjcho_work/250930/cat_and_dogs/**/*")
file = files[0]

# cat_and_dogs 데이터 없으신분들
file = "E:/cjcho_work/250930/cat_and_dogs/Cat/0.jpg"
img = cv2.imread(file)
img = cv2.resize(img,(224,224))
import matplotlib.pyplot as plt
import numpy as np 
%matplotlib auto
plt.imshow(img[...,::-1])
# 이피션트 넷 모델의 3번째 레이어의 출력을 보고 싶을 때
## model.layers[1]하는 이유는 
# model.layers[1]에 이피션트넷이 들어가 있어서
new_model = tf.keras.Model(
    model.layers[1].input, model.layers[1].layers[5].output)

new_model = tf.keras.Model(
    model.layers[1].input, model.layers[1].layers[30].output)

new_model = tf.keras.Model(
    model.layers[1].input, model.layers[1].layers[52].output)

new_model.summary()
pred = new_model.predict(img[np.newaxis])
view = pred[0,:,:,3].copy()

top_left = (10, 10)
bottom_right = (10 + 5, 10 + 5)  # width=5, height=5
# 사각형 그리기 (파랑색, 두께=1)
cv2.rectangle(view, top_left, bottom_right, 0, 1)
plt.imshow(view)
