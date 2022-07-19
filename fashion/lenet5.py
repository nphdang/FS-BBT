import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import datetime
import argparse
from sklearn.metrics import accuracy_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils import to_categorical
from keras import backend as K
import read_data
import lenet5_model

import os
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default="-1", type=str, nargs='?', help='gpu id')
parser.add_argument('--dataset', default="fashion", type=str, nargs='?', help='dataset name')
parser.add_argument('--datatype', default="original", type=str, nargs='?', help='dataset type to train teacher')
parser.add_argument('--batchsize', default="64", type=str, nargs='?', help='batch size')
parser.add_argument('--epochs', default="20", type=str, nargs='?', help='epochs')
args = parser.parse_args()
print("gpu_id: {}, dataset: {}, data_type: {}, batch_size: {}, epochs: {}".
      format(args.gpu, args.dataset, args.datatype, args.batchsize, args.epochs))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

np.random.seed(123)

dataset = args.dataset
data_type = args.datatype
batch_size = int(args.batchsize)
batch_size_predict = 256
epochs = int(args.epochs)
save_folder_model = "teacher_model"

# model name
model_type = 'lenet5'

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

# read data
X_train, y_train, X_test, y_test, n_train, n_test, n_feature, n_class, img_shape = read_data.from_file(dataset, data_type)
X_train = X_train.reshape((X_train.shape[0], img_shape[0], img_shape[1], img_shape[2]))
X_test = X_test.reshape((X_test.shape[0], img_shape[0], img_shape[1], img_shape[2]))
# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# convert labels from integers to one-hot encodings
y_train = to_categorical(y_train, n_class)
y_test = to_categorical(y_test, n_class)

teacher = lenet5_model.lenet5(input_shape=img_shape, num_classes=n_class)
teacher.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
teacher.summary()
print("model_type: {}".format(model_type))

hist = teacher.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), shuffle=True)

# plot training loss
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
fig, ax = plt.subplots(figsize=golden_size(6))
hist_df = pd.DataFrame(hist.history)
hist_df.plot(ax=ax)
ax.set_ylabel('loss')
ax.set_xlabel('# epochs')
ax.set_ylim(.99 * hist_df[1:].values.min(), 1.1 * hist_df[1:].values.max())
plt.savefig('./{}/{}_loss_{}_{}_bs{}_ep{}.pdf'.
            format(save_folder_model, model_type, dataset, data_type, batch_size, epochs), bbox_inches="tight")
plt.close()

# compute accuracy
y_pred = teacher.predict(X_test, batch_size=batch_size_predict)
y_pred_round = np.array([np.argmax(y) for y in y_pred])
# convert labels from integers to one-hot encodings
y_pred_round = to_categorical(y_pred_round, n_class)
accuracy = accuracy_score(y_test, y_pred_round)
f1_macro = f1_score(y_test, y_pred_round, average="macro")
print("teacher - accuracy: {}, f1_macro: {}".format(round(accuracy, 4), round(f1_macro, 4)))

# save model
teacher.save("./{}/{}_{}_{}_bs{}_ep{}.h5".format(save_folder_model, model_type, dataset, data_type, batch_size, epochs))
# delete model to clear memory
del teacher
K.clear_session()

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
runtime = end_time - start_time
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(runtime, 2)))

# save result to file
with open('./{}/{}_{}_{}_bs{}_ep{}.txt'.format(save_folder_model, model_type, dataset, data_type, batch_size, epochs), 'w') as f:
  f.write("dataset: {}, data_type: {}\n".format(dataset, data_type))
  f.write("teacher - accuracy: {}, f1_macro: {}\n".format(round(accuracy, 4), round(f1_macro, 4)))
  f.write("runtime: {}(s)".format(round(runtime, 2)))


