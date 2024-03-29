import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import timeit
import datetime
import argparse
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
from keras.models import load_model
import read_data

import os
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default="-1", type=str, nargs='?', help='gpu id')
parser.add_argument('--dataset', default="cifar100", type=str, nargs='?', help='dataset name')
parser.add_argument('--cvaedatatype', default="subset", type=str, nargs='?', help='dataset type to train cvae')
parser.add_argument('--cvaemethod', default="input_cond", type=str, nargs='?', help='method to train cvae')
parser.add_argument('--teacherdatatype', default="original", type=str, nargs='?', help='dataset type to train teacher')
parser.add_argument('--latent', default="2", type=str, nargs='?', help='latent')
parser.add_argument('--batchsize', default="64", type=str, nargs='?', help='batch size')
parser.add_argument('--epochs', default="600", type=str, nargs='?', help='epochs')
args = parser.parse_args()
print("gpu_id: {}, dataset: {}, cvae_data_type: {}, cvae_method: {}, "
      "latent: {}, batch_size: {}, epochs: {}, "
      "teacher_data_type: {}".
      format(args.gpu,
             args.dataset, args.cvaedatatype, args.cvaemethod,
             args.latent, args.batchsize, args.epochs,
             args.teacherdatatype))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

np.random.seed(123)

### functions for CVAE ###
# KL divergence layer
class KLDivergenceLayer(Layer):
    # identity transform layer that adds KL divergence to the final model loss
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)
    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

# Negative log likelihood (Bernoulli)
def cvae_loss(x_true, x_pred):
    # keras.losses.binary_crossentropy gives the mean over the last axis but we require the sum
    # we cannot use bc (i.e. losses.binary_crossentropy) as it is K.mean(K.binary_crossentropy())
    # while here we need K.sum(K.binary_crossentropy())
    # loss = bc(x_true_train, x_pred_train)
    # loss makes input image and reconstructed image similar
    loss = K.sum(K.binary_crossentropy(x_true, x_pred), axis=-1)

    return loss

dataset = args.dataset
cvae_data_type = args.cvaedatatype
cvae_method = args.cvaemethod
teacher_data_type = args.teacherdatatype
latent = int(args.latent)
batch_size = int(args.batchsize)
batch_size_predict = 256
epochs = int(args.epochs)
load_folder_model = "teacher_model"
save_folder_model = "generative_model"

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

# read data
X_train, _, _, _, n_train, _, n_feature, n_class, img_shape = read_data.from_file(dataset, cvae_data_type)
X_train = X_train.reshape((X_train.shape[0], img_shape[0], img_shape[1], img_shape[2]))
# normalize data
X_train = X_train.astype('float32')
X_train /= 255
# convert from matrix to vector
X_train_vector = copy.deepcopy(X_train)
X_train_vector = X_train_vector.reshape(-1, n_feature)

# load teacher model
teacher_model_type = "resnet32"
teacher_batch_size = 32
teacher_epochs = 200
print("teacher_model_type: {}".format(teacher_model_type))
teacher = load_model("./{}/{}_{}_{}_bs{}_ep{}.h5".format(load_folder_model, teacher_model_type, dataset,
                                                         teacher_data_type, teacher_batch_size, teacher_epochs))
# use teacher to predict labels for X_train
y_train_pred = teacher.predict(X_train, batch_size=batch_size_predict)
y_train_round = np.array([np.argmax(y) for y in y_train_pred])
# convert labels from integers to one-hot encodings
y_train_round = to_categorical(y_train_round, n_class)

# train CVAE
# Encoder: q(z|x)
x = Input(shape=img_shape)
cond = Input(shape=(n_class,))
if cvae_method == "input_cond":
    # convert condition from vector to image format
    cond_cnn = Dense(img_shape[0] * img_shape[1] * 1)(cond)
    cond_cnn = Reshape(target_shape=(img_shape[0], img_shape[1], 1))(cond_cnn)
    x_cond = Concatenate()([x, cond_cnn])
    q = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(x_cond)
if cvae_method == "feature_cond":
    q = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(x)
q = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(q)
q = Conv2D(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')(q)
q = Conv2D(filters=256, kernel_size=4, strides=2, padding='same', activation='relu')(q)
q = Conv2D(filters=512, kernel_size=4, strides=2, padding='same', activation='relu')(q)
# get shape of q
q_shape = K.int_shape(q)
print("shape of q: {}".format(q_shape))
flat = Flatten()(q)
if cvae_method == "input_cond":
    h_q = Dense(32, activation='relu')(flat)
if cvae_method == "feature_cond":
    flat_cond = Concatenate()([flat, cond])
    h_q = Dense(32, activation='relu')(flat_cond)
h_q = Dense(16, activation='relu')(h_q)
z_mu = Dense(latent)(h_q)
z_log_var = Dense(latent)(h_q)
z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
# equivalent to sample_z() in Keras
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
# z has a simple distribution N(0, 1)
eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent), mean=0.0, stddev=1.0))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])
z_cond = Concatenate()([z, cond])
encoder = Model([x, cond], z_mu, name="encoder")
# Decoder: p(x|z)
x_encoded = Input(shape=(latent + n_class,))
h_p = Dense(16, activation='relu')(x_encoded)
h_p = Dense(32, activation='relu')(h_p)
h = Dense(q_shape[1] * q_shape[2] * q_shape[3])(h_p)
p = Reshape(target_shape=(q_shape[1], q_shape[2], q_shape[3]))(h)
p = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same', activation='relu')(p)
p = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', activation='relu')(p)
p = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')(p)
p = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(p)
p = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(p)
p = Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', activation='relu')(p)
flat = Flatten()(p)
x_decoded = Dense(n_feature, activation='sigmoid')(flat)
decoder = Model(x_encoded, x_decoded, name="decoder")
# create model
x_pred = decoder(z_cond)
cvae = Model(inputs=[x, cond, eps], outputs=x_pred, name='cvae')
print("---Encoder summary---")
encoder.summary()
print("---Decoder summary---")
decoder.summary()
print("---CVAE summary---")
cvae.summary()

# train model
cvae.compile(optimizer='adam', loss=cvae_loss)
hist = cvae.fit([X_train, y_train_round], X_train_vector, shuffle=True, epochs=epochs, batch_size=batch_size,
                validation_split=0.05, verbose=1)
# save Encoder, Decoder
encoder.save("./{}/cvae_{}_encoder_{}_{}_method_{}_la{}_bs{}_ep{}_teacher_{}.h5".
             format(save_folder_model, teacher_model_type, dataset, cvae_data_type, cvae_method, latent, batch_size, epochs, teacher_data_type))
decoder.save("./{}/cvae_{}_decoder_{}_{}_method_{}_la{}_bs{}_ep{}_teacher_{}.h5".
             format(save_folder_model, teacher_model_type, dataset, cvae_data_type, cvae_method, latent, batch_size, epochs, teacher_data_type))

# plot training loss
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
fig, ax = plt.subplots(figsize=golden_size(6))
hist_df = pd.DataFrame(hist.history)
hist_df.plot(ax=ax)
ax.set_ylabel('loss')
ax.set_xlabel('# epochs')
ax.set_ylim(.99 * hist_df[1:].values.min(), 1.1 * hist_df[1:].values.max())
plt.savefig('./{}/cvae_{}_loss_{}_{}_method_{}_la{}_bs{}_ep{}_teacher_{}.pdf'.
            format(save_folder_model, teacher_model_type, dataset, cvae_data_type, cvae_method, latent, batch_size, epochs, teacher_data_type),
            bbox_inches="tight")
plt.close()

# evaluate model: reconstructed images
X_train_encoded = encoder.predict([X_train, y_train_round], batch_size=batch_size_predict)
X_train_encoded_y_train = np.append(X_train_encoded, y_train_round, axis=1)
X_train_decoded = decoder.predict(X_train_encoded_y_train, batch_size=batch_size_predict)

# evaluate model: generated images
z_random = norm.ppf(np.random.rand(n_train, latent))
z_random_y_train = np.append(z_random, y_train_round, axis=1)
X_random_decoded = decoder.predict(z_random_y_train, batch_size=batch_size_predict)

# plot images
img_rows, img_cols, img_chns = img_shape[0], img_shape[1], img_shape[2]
n_plot_images = 10
plt.figure(figsize=(10, 3))
for idx in range(n_plot_images):
    # plot original images
    plt.subplot(3, n_plot_images, idx + 1)
    plt.axis('off')
    plt.imshow(X_train[idx].reshape(img_rows, img_cols, img_chns))
    # plot reconstructed images
    plt.subplot(3, n_plot_images, idx + 1 + n_plot_images)
    plt.axis('off')
    plt.imshow(X_train_decoded[idx].reshape(img_rows, img_cols, img_chns))
    # plot generated images
    plt.subplot(3, n_plot_images, idx + 1 + n_plot_images + n_plot_images)
    plt.axis('off')
    plt.imshow(X_random_decoded[idx].reshape(img_rows, img_cols, img_chns))
plt.suptitle("first row: original images, second row: reconstructed images, third row: generated images")
plt.savefig('./{}/cvae_{}_images_{}_{}_method_{}_la{}_bs{}_ep{}_teacher_{}.pdf'.
            format(save_folder_model, teacher_model_type, dataset, cvae_data_type, cvae_method, latent, batch_size, epochs, teacher_data_type),
            bbox_inches="tight")
plt.close()

# delete model to clear memory
del encoder, decoder, cvae
K.clear_session()

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
runtime = end_time - start_time
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(runtime, 2)))

# save result to file
with open('./{}/cvae_{}_{}_{}_method_{}_la{}_bs{}_ep{}_teacher_{}.txt'.
                  format(save_folder_model, teacher_model_type, dataset, cvae_data_type, cvae_method, latent, batch_size, epochs, teacher_data_type), 'w') as f:
  f.write("dataset: {}, cvae_data_type: {}, cvae_method: {}\n".format(dataset, cvae_data_type, cvae_method))
  f.write("teacher_data_type: {}\n".format(teacher_data_type))
  f.write("runtime: {}(s)".format(round(runtime, 2)))

