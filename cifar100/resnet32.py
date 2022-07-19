import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import datetime
import argparse
from sklearn.metrics import accuracy_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras import optimizers
from keras.utils import to_categorical
from keras import backend as K
import read_data
import resnet_model

import os
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default="-1", type=str, nargs='?', help='gpu id')
parser.add_argument('--dataset', default="cifar100", type=str, nargs='?', help='dataset name')
parser.add_argument('--datatype', default="original", type=str, nargs='?', help='dataset type to train teacher')
parser.add_argument('--batchsize', default="32", type=str, nargs='?', help='batch size')
parser.add_argument('--epochs', default="200", type=str, nargs='?', help='epochs')
parser.add_argument('--augmentation', default="True", type=str, nargs='?', help='data augmentation')
args = parser.parse_args()
print("gpu_id: {}, dataset: {}, data_type: {}, batch_size: {}, epochs: {}, data_augmentation: {}".
      format(args.gpu, args.dataset, args.datatype, args.batchsize, args.epochs, args.augmentation))

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
data_augmentation = bool(args.augmentation)
save_folder_model = "teacher_model"

# model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

n = 5
# compute depth from supplied model parameter n
depth = n * 6 + 2

# model name and depth
model_type = 'resnet{}'.format(depth)

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

# read data
X_train, y_train, X_test, y_test, _, _, _, n_class, img_shape = read_data.from_file(dataset, data_type)
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

# use learning rate decay
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


teacher = resnet_model.resnet_v1(input_shape=img_shape, depth=depth, num_classes=n_class)
teacher.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr_schedule(0)), metrics=['accuracy'])
teacher.summary()
print("model_type: {}".format(model_type))

# prepare callbacks for learning rate adjustment
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [lr_reducer, lr_scheduler]

if data_augmentation == False:
    print('Not using data augmentation.')
    hist = teacher.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
                       shuffle=True, callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True
        )
    datagen.fit(X_train)
    hist = teacher.fit_generator(datagen.flow(X_train, y_train, shuffle=True, batch_size=batch_size),
                                 epochs=epochs, verbose=1, validation_data=(X_test, y_test), workers=4, callbacks=callbacks)

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


