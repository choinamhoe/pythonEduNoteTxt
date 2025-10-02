import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib auto

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train.shape
x_test.shape

plt.imshow(x_train[0])

x_train = x_train/255.
x_test = x_test/255.

# train 은 6만장 test는 1만장
x_train_flat = x_train.reshape(60000,28*28)
x_test_flat = x_test.reshape(10000,28*28)

### 간단한 오토 인코더 구현하기
inp = tf.keras.layers.Input(shape=(784,))
x = tf.keras.layers.Dense(128, activation="relu")(inp)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)

x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
out = tf.keras.layers.Dense(784, activation="sigmoid")(x)

auto_encoder_model = tf.keras.Model(inp, out)
auto_encoder_model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy"
    )
auto_encoder_model.fit(
    x_train_flat, x_train_flat,
    epochs = 20,batch_size = 32
    )

### 40분까지 정리 
# 모델 레이어에서 
# 차원이 축소되는 부분(encoder)과 차원이 복원되는 부분(decoder)
# 각각 모델 2개로 만들어보기 # 45분
auto_encoder_model.summary()

auto_encoder_model.layers
auto_encoder_model.input
auto_encoder_model.output
auto_encoder_model.summary()


encoder = tf.keras.Model(
    auto_encoder_model.input,
    auto_encoder_model.layers[3].output
    )
encoder.summary()

auto_encoder_model.layers[4].input
#auto_encoder_model.layers[-1].output
auto_encoder_model.output

decoder = tf.keras.Model(
    auto_encoder_model.layers[4].input,
    auto_encoder_model.output
    )
decoder.summary()

encoder_results = encoder.predict(x_test_flat)
encoder_results.shape
decoder_results = decoder.predict(encoder_results)
decoder_results.shape

# 원본 이미지와 오토인코더 결과 같이 시각화
n=20
auto_encoder_result = decoder_results[n].reshape(28, 28)

origin = x_test_flat[n].reshape(28,28)
auto_encoder_result
view = np.concatenate([origin, auto_encoder_result])
plt.imshow(view)

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255. # 범위 0~1 사이로 변형
x_test = x_test/255.

# 정규분포 활용해서 노이즈 생성
tr_noise = np.random.normal(loc=0.0, scale=1.0, size = x_train.shape)
te_noise = np.random.normal(loc=0.0, scale=1.0, size = x_test.shape)

### 노이즈가 영향이 너무 커서 3으로 나눔
x_train_noise = x_train + tr_noise/3
x_test_noise = x_test+ te_noise/3
plt.imshow(x_train_noise[1])
# 노이즈를 더해줬기 때문에 범위가 1을 넘을 수 있어서 범위를 0~1사이로 변경
x_train_noise = np.clip(x_train_noise, 0., 1.)
x_test_noise = np.clip(x_test_noise, 0., 1.)

# conv > Bn (정규화) > Activation(활성화함수) > MaxPool(풀링) 순
inp = tf.keras.layers.Input(shape=(28,28, 1))
x = tf.keras.layers.Conv2D(
    32, (3, 3), padding="same", activation="relu")(inp)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(
    64, (3, 3), padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(
    64, (3, 3), padding="same", activation="relu")(x)
x = tf.keras.layers.UpSampling2D((2,2))(x)
x = tf.keras.layers.Conv2D(
    32, (3, 3), padding="same", activation="relu")(x)
x = tf.keras.layers.UpSampling2D((2,2))(x)
out = tf.keras.layers.Conv2D(
    1, (3, 3), padding="same", activation="sigmoid")(x)
auto_encoder = tf.keras.Model(inp, out)
auto_encoder.summary()

# UpSample2D 대신 Conv2DTranspose레이어도 사용이 가능
## Conv2DTranspose 사용하면
## Upsampling2D + Conv2D -> Conv2DTranspose 형태
# tf.keras.layers.Conv2DTranspose(64, (3, 3), strides = 2, padding="same")
auto_encoder.compile(
    optimizer="adam", loss="binary_crossentropy")
auto_encoder.fit(
    x_train_noise,x_train,
    epochs = 100,
    batch_size = 32
    )

auto_encoder.summary()
encoder = tf.keras.Model(
    auto_encoder.input,
    auto_encoder.layers[5].output
    )
decoder = tf.keras.Model(
    auto_encoder.layers[6].input,
    auto_encoder.output
    )

encoder_result = encoder.predict(x_test_noise)
decoder_result = decoder.predict(encoder_result)

n = 10
view_data = np.concatenate(
    [
         x_test_noise[n],
         decoder_result[n,:,:,0],
         x_test[n]
     ]
    )
plt.imshow(view_data)

inp = tf.keras.layers.Input(shape = (784,))
h = tf.keras.layers.Dense(128, activation="relu")(inp)
# 축소할 차원의 수(잠재공간의 크기)
z_mean = tf.keras.layers.Dense(2)(h)
z_log_var = tf.keras.layers.Dense(2)(h)

def sampling(args):
    # 표준 정규분포를 활용해서 데이터를 랜덤하게 샘플링하는 코드 
    z_mean, z_log_var = args
    # 렌덤 부분
    epsilon = tf.keras.backend.random_normal(
        shape = (
            tf.keras.backend.shape(z_mean)[0],
            2 # 잠재공간 크기 
        )
        )
    # 표준정규분포의 평균 + 표준편차* 노이즈
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

z = tf.keras.layers.Lambda(
    sampling, output_shape= (2,))([z_mean, z_log_var])
encoder = tf.keras.Model(inp, [z_mean, z_log_var, z])

dec_inp = tf.keras.layers.Input(shape=(2,))# 잠재공간의 크기와 동일
x = tf.keras.layers.Dense(128, activation="relu")(dec_inp)
x = tf.keras.layers.Dense(128, activation="relu")(x)
dec_out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
decoder = tf.keras.Model(dec_inp, dec_out)

out = decoder(z)
vae = tf.keras.Model(inp, out)
vae.summary()

reconstruction_loss = tf.keras.losses.binary_crossentropy(inp, out) * 2

kl_loss = 1 + z_log_var - z_mean**2 - tf.keras.backend.exp(z_log_var)
kl_loss = -0.5 * tf.keras.backend.sum(kl_loss, axis=-1)
vae_loss = tf.keras.backend.mean(kl_loss + reconstruction_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

input_data=np.array([[1,1]])
pred_img = decoder.predict(input_data)
plt.imshow(pred_img[0].reshape(28,28))