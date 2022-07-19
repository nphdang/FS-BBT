import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import timeit
import datetime
import argparse
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import norm
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils import to_categorical
from keras import backend as K
from keras.models import load_model
from keras.losses import categorical_crossentropy as ce
from keras.losses import kullback_leibler_divergence as kl
import read_data
import lenet5_model
from mixup_generator import MixupGenerator

import os
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default="-1", type=str, nargs='?', help='gpu id')
parser.add_argument('--dataset', default="fashion", type=str, nargs='?', help='dataset name')
parser.add_argument('--studentdatatype', default="subset", type=str, nargs='?', help='dataset type to train student')
parser.add_argument('--teacherdatatype', default="original", type=str, nargs='?', help='dataset type when training teacher')
parser.add_argument('--cvaedatatype', default="subset", type=str, nargs='?', help='dataset type when training cvae')
parser.add_argument('--cvaelatent', default="2", type=str, nargs='?', help='latent when training cvae')
parser.add_argument('--cvaebatchsize', default="256", type=str, nargs='?', help='batch size when training cvae')
parser.add_argument('--cvaeepochs', default="100", type=str, nargs='?', help='epochs when training cvae')
parser.add_argument('--balance', default="0.5", type=str, nargs='?', help='trade-off factor between teacher predictions and true labels')
parser.add_argument('--batchsize', default="64", type=str, nargs='?', help='batch size to train student')
parser.add_argument('--epochs', default="20", type=str, nargs='?', help='epochs to train student')
parser.add_argument('--augmentation', default="standard", type=str, nargs='?', help='type of data augmentation')
parser.add_argument('--mxalpha', default="0.2", type=str, nargs='?', help='mixup trade-off factor')
parser.add_argument('--nogen', default="40000", type=str, nargs='?', help='no of generated images')
parser.add_argument('--sampling', default="gaussian", type=str, nargs='?', help='how to sample z')
parser.add_argument('--nomixup', default="0", type=str, nargs='?', help='no of mixup images')
parser.add_argument('--loss', default="ce", type=str, nargs='?', help='type of loss function')
parser.add_argument('--round', default="2", type=str, nargs='?', help='rounding number')
parser.add_argument('--threshold', default="1", type=str, nargs='?', help='threshold to determine non-mixup images')
parser.add_argument('--run', default="1", type=str, nargs='?', help='run id')

args = parser.parse_args()
print("gpu_id: {}, "
      "dataset: {}, student_data_type: {}, teacher_data_type: {}, "
      "cvae_data_type: {}, cvae_latent: {}, cvae_batch_size: {}, cvae_epochs: {}, "      
      "balance: {}, batch_size: {}, epochs: {}, "
      "data_augmentation: {}, mx_alpha: {}, loss: {}, "
      "number_generate: {}, number_mixup: {}".
      format(args.gpu,
             args.dataset, args.studentdatatype, args.teacherdatatype,
             args.cvaedatatype, args.cvaelatent, args.cvaebatchsize, args.cvaeepochs,
             args.balance, args.batchsize, args.epochs,
             args.augmentation, args.mxalpha, args.loss,
             args.nogen, args.nomixup))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

run_id = int(args.run)
np.random.seed(run_id)

dataset = args.dataset
student_data_type = args.studentdatatype
teacher_data_type = args.teacherdatatype
cvae_data_type = args.cvaedatatype
cvae_latent = int(args.cvaelatent)
cvae_batch_size = int(args.cvaebatchsize)
cvae_epochs = int(args.cvaeepochs)
balance = float(args.balance)
batch_size = int(args.batchsize)
batch_size_predict = 256
epochs = int(args.epochs)
data_augmentation = args.augmentation # none, standard, mixup, combination
mx_alpha = float(args.mxalpha)
loss_type = args.loss # ce, kd
number_generate = int(args.nogen) # -1: don't use generated images (BBKD) and 0: use generated images
sampling = args.sampling
number_mixup = int(args.nomixup)
rounding_numer = int(args.round)
threshold = float(args.threshold)
load_teacher_model = "teacher_model"
load_reconstructed_distilled_data = "generative_model"
save_folder_model = "student_model"

# model name and depth
student_model_type = 'lenet5_light_kd'

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

# read data
X_train, y_train_org, X_test, y_test, n_train, n_test, n_feature, n_class, img_shape = read_data.from_file(dataset, student_data_type)
X_train = X_train.reshape((X_train.shape[0], img_shape[0], img_shape[1], img_shape[2]))
X_test = X_test.reshape((X_test.shape[0], img_shape[0], img_shape[1], img_shape[2]))
# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# convert labels from integers to one-hot encodings
y_train_org = to_categorical(y_train_org, n_class)
y_test = to_categorical(y_test, n_class)

# load teacher model
teacher_model_type = "lenet5"
teacher_batch_size = 64
teacher_epochs = 20
print("teacher_model_type: {}".format(teacher_model_type))
teacher = load_model("./{}/{}_{}_{}_bs{}_ep{}.h5".format(load_teacher_model, teacher_model_type, dataset,
                                                         teacher_data_type, teacher_batch_size, teacher_epochs))
print("---Teacher summary---")
print(teacher.summary())
# use teacher to predict labels for X_train
y_train_pred = teacher.predict(X_train, batch_size=batch_size_predict)
y_train_round = np.array([np.argmax(y) for y in y_train_pred])
# convert labels from integers to one-hot encodings
y_train_round = to_categorical(y_train_round, n_class)
accuracy_teacher = accuracy_score(y_train_org, y_train_round)
f1_macro_teacher = f1_score(y_train_org, y_train_round, average="macro")
print("teacher - train accuracy: {}, f1_macro: {}".format(round(accuracy_teacher, 4), round(f1_macro_teacher, 4)))

# generate mixup images
if number_mixup > 0:
    factor_mixup = int(number_mixup / n_train)
    X_train_mx = copy.deepcopy(X_train)
    y_train_mx = copy.deepcopy(y_train_round)
    for _ in range(factor_mixup - 1):
        X_train_mx = np.append(X_train_mx, X_train, axis=0)
        y_train_mx = np.append(y_train_mx, y_train_round, axis=0)
    # create two random training sets
    train_indices = np.arange(number_mixup)
    np.random.shuffle(train_indices)
    X_train_1 = copy.deepcopy(X_train_mx[train_indices])
    y_train_1 = copy.deepcopy(y_train_mx[train_indices])
    np.random.shuffle(train_indices)
    X_train_2 = copy.deepcopy(X_train_mx[train_indices])
    y_train_2 = copy.deepcopy(y_train_mx[train_indices])
    print("X_train_1: {}, y_train_1: {}".format(X_train_1.shape, y_train_1.shape))
    print("X_train_2: {}, y_train_2: {}".format(X_train_2.shape, y_train_2.shape))

    # mixup method
    def mix_up(X1, y1, X2, y2, alpha=0.2):
        # unpack two datasets
        images_one, labels_one = X1, y1
        images_two, labels_two = X2, y2

        # sample lambda from Beta distribution
        n_sample = images_one.shape[0]
        l = np.random.beta(alpha, alpha, size=n_sample)
        # remove mixup images with lambda = 0 or lambda =1 as they are original images
        l = np.around(l, rounding_numer)
        # threshold must be small value: threshold <= 0.1
        indices_not_mixup = np.logical_or(l <= threshold, l >= (1 - threshold))
        # print("n_sample: {}, l: {}".format(n_sample, np.around(l, 2)))
        # reshape to match lambda with image and label
        x_l = l.reshape(n_sample, 1, 1, 1) # three channels
        y_l = l.reshape(n_sample, 1) # one dimension

        # perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)

        return images, labels, indices_not_mixup


    # create the new training dataset using mixup method
    X_train_mu, y_train_mu, indices_not_mixup = mix_up(X_train_1, y_train_1, X_train_2, y_train_2, alpha=mx_alpha)
    print("no of mixup images: {}".format(X_train_mu.shape))
    if number_generate == 0:
        # use generated images for original images
        number_generate = len(X_train_mu[indices_not_mixup])
        print("no of original images: {}".format(number_generate))
        # get mixup images not original images
        X_train_mu = X_train_mu[~indices_not_mixup]
        y_train_mu = y_train_mu[~indices_not_mixup]
        print("no of mixup images excluding original images: {}".format(X_train_mu.shape))

    # use teacher to predict labels for X_train_mu
    y_mu_pred = teacher.predict(X_train_mu, batch_size=batch_size_predict)

# use generated data to train student
if number_generate > 0:
    # load decoder of CVAE
    decoder = load_model("./{}/cvae_decoder_{}_{}_la{}_bs{}_ep{}_teacher_{}.h5".
                         format(load_reconstructed_distilled_data, dataset,
                                cvae_data_type, cvae_latent, cvae_batch_size, cvae_epochs,
                                teacher_data_type))
    if sampling == "gaussian":
        z_random = np.random.normal(size=(number_generate, cvae_latent))
    if sampling == "uniform":
        z_uniform_1 = np.random.uniform(-3, 3, size=number_generate)
        z_uniform_2 = np.random.uniform(-3, 3, size=number_generate)
        z_random = np.array(list(zip(z_uniform_1, z_uniform_2)))
    if sampling == "hybrid":
        number_generate_half = int(number_generate / 2)
        z_gaussian = np.random.normal(size=(number_generate_half, cvae_latent))
        z_uniform_1 = np.random.uniform(-3, 3, size=number_generate_half)
        z_uniform_2 = np.random.uniform(-3, 3, size=number_generate_half)
        z_uniform = np.array(list(zip(z_uniform_1, z_uniform_2)))
        z_random = np.append(z_gaussian, z_uniform, axis=0)
        # update number_generate
        number_generate = z_random.shape[0]
    print("no of generated images (z_cvae): {}".format(z_random.shape))

    # transform z_random with mu and sigma
    mu_sigma = [0, 0, 1, 1]
    mu = mu_sigma[:cvae_latent]
    mu_mx = np.array([mu] * number_generate)
    sigma = mu_sigma[cvae_latent:]
    sigma_mx = np.zeros([cvae_latent, cvae_latent])
    for sig_idx, sig_val in enumerate(sigma):
        sigma_mx[sig_idx, sig_idx] = sig_val
    z_random_generate = mu_mx + np.matmul(z_random, sigma_mx)
    # compute no of generated samples for each class
    n_gen_each_class = int(number_generate / n_class)
    y_cvae = []
    for label in range(n_class):
        y_cvae = np.append(y_cvae, np.array([label] * n_gen_each_class), axis=0)
    # convert labels from integers to one-hot encodings
    y_cvae = to_categorical(y_cvae, n_class)
    print("no of generated images (y_cvae): {}".format(y_cvae.shape))
    # need to check if length of z_cvae and y_cvae is the same
    len_z_cvae = len(z_random_generate)
    len_y_cvae = len(y_cvae)
    print("len_z_cvae: {}, len_y_cvae: {}".format(len_z_cvae, len_y_cvae))
    if len_z_cvae > len_y_cvae:
        z_random_generate = z_random_generate[:len_y_cvae]
    if len_y_cvae > len_z_cvae:
        y_cvae = y_cvae[:len_z_cvae]
    z_random_y_cvae = np.append(z_random_generate, y_cvae, axis=1)
    X_cvae = decoder.predict(z_random_y_cvae, batch_size=batch_size_predict)
    X_cvae = X_cvae[:, :n_feature]
    # reshape to RGB format
    X_cvae = X_cvae.reshape((X_cvae.shape[0], img_shape[0], img_shape[1], img_shape[2]))
    """
    # plot generated images
    n_plot_images = 20
    plt.figure(figsize=(n_plot_images, 2))
    for idx in range(n_plot_images):
        # plot original images
        plt.subplot(2, n_plot_images, idx + 1)
        plt.axis('off')
        plt.imshow(X_cvae[idx].reshape(img_shape))
        # plot modified images
        plt.subplot(2, n_plot_images, idx + 1 + n_plot_images)
        plt.axis('off')
        plt.imshow(X_cvae[idx + 1 + n_plot_images].reshape(img_shape))
    plt.suptitle("generated images")
    plt.savefig('./generated_images_{}.pdf'.format(sampling), bbox_inches="tight")
    plt.close()
    """
    # use teacher to predict labels for X_cvae
    y_cvae_pred = teacher.predict(X_cvae, batch_size=batch_size_predict)
    y_cvae_round = np.array([np.argmax(y) for y in y_cvae_pred])
    # convert labels from integers to one-hot encodings
    y_cvae_round = to_categorical(y_cvae_round, n_class)

    # NOTE: y_cvae are fixed labels used to generate images X_cvae
    #       y_cvae_round are predicted labels of teacher for X_cvae

# create training images
X_ce = copy.deepcopy(X_train)
if number_generate > 0:
    # concat original images and generated images
    X_ce = np.append(X_ce, X_cvae, axis=0)
if number_mixup > 0:
    # concat original images and mixup images
    X_ce = np.append(X_ce, X_train_mu, axis=0)
# create true labels
y_ce = copy.deepcopy(y_train_round)
if number_generate > 0:
    y_ce = np.append(y_ce, y_cvae, axis=0)
if number_mixup > 0:
    y_ce = np.append(y_ce, y_train_mu, axis=0)
# create teacher predictions
y_kl = copy.deepcopy(y_train_pred)
if number_generate > 0:
    y_kl = np.append(y_kl, y_cvae_pred, axis=0)
if number_mixup > 0:
    y_kl = np.append(y_kl, y_mu_pred, axis=0)
# concatenate y_ce and y_kl
y_ce_y_kl = np.concatenate([y_ce, y_kl], axis=1)

# concatenate y_test and y_teacher to use validation_data on training model
y_test_pred = teacher.predict(X_test, batch_size=batch_size_predict)
y_test_y_teacher = np.concatenate([y_test, y_test_pred], axis=1)

# train student model with knowledge distillation
# define kd loss
def kd_loss(y_true, y_pred, alpha=0.5):
    alpha = balance
    print("alpha: {}".format(alpha))
    # y_true contains 2 parts: (1) true labels of training data and (2) predictions of teacher for training data
    y_train, y_teacher = y_true[:, :n_class], y_true[:, n_class:]
    y_student = y_pred
    loss = alpha * ce(y_train, y_student) + (1 - alpha) * kl(y_teacher, y_student)

    return loss

student = lenet5_model.lenet5(input_shape=img_shape, num_classes=n_class)
if loss_type == "ce":
    student.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
if loss_type == "kd":
    student.compile(loss=kd_loss, optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
print("---Student summary---")
print(student.summary())
print("student_model_type: {}".format(student_model_type))

print("X_ce: {}".format(X_ce.shape))
if data_augmentation == "none":
    print('Using none data augmentation.')
    if loss_type == "ce":
        """
        hist = student.fit(X_ce, y_ce, batch_size=batch_size, epochs=epochs,
                           validation_data=(X_test, y_test), shuffle=True)
        """
        hist = student.fit(X_ce, y_kl, batch_size=batch_size, epochs=epochs,
                           validation_data=(X_test, y_test), shuffle=True)
    if loss_type == "kd":
        hist = student.fit(X_ce, y_ce_y_kl, batch_size=batch_size, epochs=epochs,
                           validation_data=(X_test, y_test_y_teacher), shuffle=True)
if data_augmentation == "standard":
    print('Using standard data augmentation.')
    datagen = ImageDataGenerator(
        # randomly rotate images
        rotation_range=10,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1
        )
    datagen.fit(X_ce)
    if loss_type == "ce":
        """
        hist = student.fit_generator(datagen.flow(X_ce, y_ce, shuffle=True, batch_size=batch_size),
                                     epochs=epochs, verbose=1, validation_data=(X_test, y_test), workers=4)
        """
        hist = student.fit_generator(datagen.flow(X_ce, y_kl, shuffle=True, batch_size=batch_size),
                                     epochs=epochs, verbose=1, validation_data=(X_test, y_test), workers=4)
    if loss_type == "kd":
        hist = student.fit_generator(datagen.flow(X_ce, y_ce_y_kl, shuffle=True, batch_size=batch_size),
                                     epochs=epochs, verbose=1, validation_data=(X_test, y_test_y_teacher), workers=4)

if data_augmentation == "mixup":
    print('Using mixup data augmentation.')
    if loss_type == "ce":
        # mxgen = MixupGenerator(X_ce, y_ce, batch_size=batch_size, alpha=mx_alpha)()
        mxgen = MixupGenerator(X_ce, y_kl, batch_size=batch_size, alpha=mx_alpha)()
        hist = student.fit_generator(generator=mxgen, steps_per_epoch=X_ce.shape[0] // batch_size,
                                     epochs=epochs, verbose=1, validation_data=(X_test, y_test))
    if loss_type == "kd":
        mxgen = MixupGenerator(X_ce, y_ce_y_kl, batch_size=batch_size, alpha=mx_alpha)()
        hist = student.fit_generator(generator=mxgen, steps_per_epoch=X_ce.shape[0] // batch_size,
                                     epochs=epochs, verbose=1, validation_data=(X_test, y_test_y_teacher))
if data_augmentation == "combination":
    print('Using combined data augmentation.')
    datagen = ImageDataGenerator(
        # randomly rotate images
        rotation_range=10,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1
    )
    if loss_type == "ce":
        # datamxgen = MixupGenerator(X_ce, y_ce, batch_size=batch_size, alpha=mx_alpha, datagen=datagen)()
        datamxgen = MixupGenerator(X_ce, y_kl, batch_size=batch_size, alpha=mx_alpha, datagen=datagen)()
        hist = student.fit_generator(generator=datamxgen, steps_per_epoch=X_ce.shape[0] // batch_size,
                                     epochs=epochs, verbose=1, validation_data=(X_test, y_test))
    if loss_type == "kd":
        datamxgen = MixupGenerator(X_ce, y_ce_y_kl, batch_size=batch_size, alpha=mx_alpha, datagen=datagen)()
        hist = student.fit_generator(generator=datamxgen, steps_per_epoch=X_ce.shape[0] // batch_size,
                                     epochs=epochs, verbose=1, validation_data=(X_test, y_test_y_teacher))

# plot training loss
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
fig, ax = plt.subplots(figsize=golden_size(6))
hist_df = pd.DataFrame(hist.history)
hist_df.plot(ax=ax)
ax.set_ylabel('loss')
ax.set_xlabel('# epochs')
ax.set_ylim(.99 * hist_df[1:].values.min(), 1.1 * hist_df[1:].values.max())
save_path = './{}/{}_loss_{}_{}_cvae_{}_bl{}_bs{}_ep{}_teacher_{}_' \
            'gen{}_sample_{}_aug_{}_alpha{}_mx{}_loss_{}_round{}_threshold{}_run{}.pdf'. \
    format(save_folder_model, student_model_type, dataset, student_data_type,
           cvae_data_type,
           balance, batch_size, epochs,
           teacher_data_type, number_generate, sampling,
           data_augmentation, mx_alpha, number_mixup,
           loss_type, rounding_numer, threshold, run_id)
plt.savefig(save_path, bbox_inches="tight")
plt.close()

# compute accuracy
y_pred_student = student.predict(X_test, batch_size=batch_size_predict)
y_pred_student_round = np.array([np.argmax(y) for y in y_pred_student])
# convert labels from integers to one-hot encodings
y_pred_student_round = to_categorical(y_pred_student_round, n_class)
accuracy_student = accuracy_score(y_test, y_pred_student_round)
f1_macro_student = f1_score(y_test, y_pred_student_round, average="macro")
print("student_light_kd - accuracy: {}, f1_macro: {}".format(round(accuracy_student, 4), round(f1_macro_student, 4)))

# save model
save_path = "./{}/{}_{}_{}_cvae_{}_bl{}_bs{}_ep{}_teacher_{}_" \
            "gen{}_sample_{}_aug_{}_alpha{}_mx{}_loss_{}_round{}_threshold{}_run{}.h5". \
    format(save_folder_model, student_model_type, dataset, student_data_type,
           cvae_data_type,
           balance, batch_size, epochs,
           teacher_data_type, number_generate, sampling,
           data_augmentation, mx_alpha, number_mixup,
           loss_type, rounding_numer, threshold, run_id)
student.save(save_path)

# delete model to clear memory
del teacher, student
K.clear_session()

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
runtime = end_time - start_time
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(runtime, 2)))

# save result to file
save_path = "./{}/{}_{}_{}_cvae_{}_bl{}_bs{}_ep{}_teacher_{}_" \
            "gen{}_sample_{}_aug_{}_alpha{}_mx{}_loss_{}_round{}_threshold{}_run{}.txt". \
    format(save_folder_model, student_model_type, dataset, student_data_type,
           cvae_data_type,
           balance, batch_size, epochs,
           teacher_data_type, number_generate, sampling,
           data_augmentation, mx_alpha, number_mixup,
           loss_type, rounding_numer, threshold, run_id)
with open(save_path, 'w') as f:
  f.write("dataset: {}, student_data_type: {}\n".format(dataset, student_data_type))
  f.write("cvae_data_type: {}, cvae_latent: {}, cvae_batch_size: {}, cvae_epochs: {}\n".
          format(cvae_data_type, cvae_latent, cvae_batch_size, cvae_epochs))
  f.write("teacher_data_type: {}\n".format(teacher_data_type))
  f.write("number_generate: {}, sampling: {}\n".format(number_generate, sampling))
  f.write("data_augmentation: {}\n".format(data_augmentation))
  f.write("number_mixup: {}, mx_alpha: {}, loss_type: {}\n".format(number_mixup, mx_alpha, loss_type))
  f.write("teacher - train accuracy: {}, f1_macro: {}\n".format(round(accuracy_teacher, 4), round(f1_macro_teacher, 4)))
  f.write("student_light_kd - accuracy: {}, f1_macro: {}\n".format(round(accuracy_student, 4), round(f1_macro_student, 4)))
  f.write("runtime: {}(s)".format(round(runtime, 2)))


