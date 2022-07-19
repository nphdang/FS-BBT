from keras.layers import Conv2D, BatchNormalization, Activation,MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten


def alexnet(input_shape, num_classes):
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=48, input_shape=input_shape, kernel_size=(5, 5), strides=(1, 1),
                         padding='same', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1),
                         padding='same', bias_initializer='ones'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', bias_initializer='ones'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', bias_initializer='ones'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # Passing it to a Fully Connected layer
        model.add(Flatten())

        # 1st Fully Connected Layer
        model.add(Dense(512, bias_initializer='zeros'))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        # 2nd Fully Connected Layer
        model.add(Dense(256, bias_initializer='zeros'))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(num_classes, bias_initializer='zeros'))
        model.add(Activation('softmax'))
        return model

def alexnet_half(input_shape, num_classes):
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=24, input_shape=input_shape, kernel_size=(5, 5), strides=(1, 1),
                         padding='same', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                         padding='same', bias_initializer='ones'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', bias_initializer='ones'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                         padding='same', bias_initializer='ones'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # Passing it to a Fully Connected layer
        model.add(Flatten())

        # 1st Fully Connected Layer
        model.add(Dense(256, bias_initializer='zeros'))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        # 2nd Fully Connected Layer
        model.add(Dense(128, bias_initializer='zeros'))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(num_classes, bias_initializer='zeros'))
        model.add(Activation('softmax'))
        return model

