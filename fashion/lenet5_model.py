from keras.layers import Conv2D, BatchNormalization, Activation,MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten


def lenet5(input_shape, num_classes):
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=6, input_shape=input_shape, kernel_size=(5, 5), strides=(1, 1),
                         padding='valid', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),
                         padding='same', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())

        # 1st Fully Connected Layer
        model.add(Dense(120, bias_initializer='zeros'))
        model.add(Activation('relu'))

        # 2nd Fully Connected Layer
        model.add(Dense(84, bias_initializer='zeros'))
        model.add(Activation('relu'))

        # Output Layer
        model.add(Dense(num_classes, bias_initializer='zeros'))
        model.add(Activation('softmax'))
        return model

def lenet5_half(input_shape, num_classes):
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=3, input_shape=input_shape, kernel_size=(5, 5), strides=(1, 1),
                         padding='valid', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1),
                         padding='same', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())

        # 1st Fully Connected Layer
        model.add(Dense(120, bias_initializer='zeros'))
        model.add(Activation('relu'))

        # 2nd Fully Connected Layer
        model.add(Dense(84, bias_initializer='zeros'))
        model.add(Activation('relu'))

        # Output Layer
        model.add(Dense(num_classes, bias_initializer='zeros'))
        model.add(Activation('softmax'))
        return model

