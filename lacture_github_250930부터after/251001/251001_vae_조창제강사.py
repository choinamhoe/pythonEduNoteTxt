# 변분 오토인코더(Variational auto encoder, VAE)


import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib auto

(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

x_train_flat = x_train.reshape(60000, 28*28)
x_test_flat = x_test.reshape(10000, 28*28)

x_train_flat.shape, x_test_flat.shape


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
"""
z = tf.keras.layers.Lambda(
    sampling, output_shape= (2,))([z_mean, z_log_var])
# z 의 출력은 잠재공간의 크기와 동일, 
# 잠재공간의 범위를 표준정규분포형태로 주기 위함
encoder = tf.keras.Model(inp, [z_mean, z_log_var, z])
#### 24분
dec_inp = tf.keras.layers.Input(shape=(2,))# 잠재공간의 크기와 동일
x = tf.keras.layers.Dense(128, activation="relu")(dec_inp)
x = tf.keras.layers.Dense(128, activation="relu")(x)
dec_out = tf.keras.layers.Dense(784, activation="sigmoid")(x)
decoder = tf.keras.Model(dec_inp, dec_out)

out = decoder(z)

vae = tf.keras.Model(inp, out)
vae.summary()
# 잠재공간의 크기만큼 곱해줌
reconstruction_loss = tf.keras.losses.binary_crossentropy(inp, out)
reconstruction_loss *= 2
kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
kl_loss = -0.5 * tf.keras.backend.sum(kl_loss, axis=-1)
vae_loss = tf.keras.backend.mean(kl_loss + reconstruction_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
vae.fit(x_train_flat, epochs=20, batch_size= 32)
"""
input_dim = 784
latent_dim = 2
# 인코더
inputs = tf.keras.layers.Input(shape=(input_dim,))
h = tf.keras.layers.Dense(128, activation='relu')(inputs)
z_mean = tf.keras.layers.Dense(latent_dim)(h)
z_log_var = tf.keras.layers.Dense(latent_dim)(h)
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var]) 
# z는 평균, 표준편차 에서 샘플링한 값
encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder') 
# 평균, 표준편차, 잠재 공간에서의 샘플

# 디코더
decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(128, activation='relu')(decoder_input)
x = tf.keras.layers.Dense(128, activation='relu')(x)
decoder_output = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
decoder = tf.keras.Model(decoder_input, decoder_output, name='decoder')

# VAE 모델
outputs = decoder(z)
vae = tf.keras.Model(inputs, outputs, name='vae')

reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= 2
kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
kl_loss = -0.5 * tf.keras.backend.sum(kl_loss, axis=-1)
vae_loss = tf.keras.backend.mean(kl_loss + reconstruction_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
vae.fit(x_train_flat,x_train_flat, epochs=20, batch_size= 32)
