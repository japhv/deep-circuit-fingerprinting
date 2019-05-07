"""
author: Toboure Gambo
author: Japheth Adhavan

This file holds the different models. A lot of code here has been reused from
the Website fingerprinting homework.
"""
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.layers import SimpleRNN
from keras.optimizers import SGD, Adamax
from keras.callbacks import TensorBoard, EarlyStopping
from keras.regularizers import l2
from keras.layers import Embedding


class RNNModel(object):

    def __init__(self, num_features, num_classes):
        model = Sequential()
        model.add(SimpleRNN(256, input_shape=(num_features, 1)))
        model.add(Dense(num_classes, activation='softmax'))

        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
            'accuracy'])
        self.model = model

    def fit(self, X_train, Y_train, batch_size, epochs, verbose):
        """
        :param X_train: a numpy ndarray of dimension (k x n) containing
            training data
        :param Y_train: a numpy ndarray of dimension (k x 1) containing
            labels for X_train
        :param batch_size: batch size to use for training
        :param epochs: number of epochs for training
        :param verbose: Console print options for training progress.
            0 - silent mode,
            1 - progress bar,
            2 - one line per epoch
        :return: None

        This method start training the model with the given data. The
        training options are configured with tensorboard and early stopping
        callbacks.

        Tensorboard could be launched by navigating to the directory
        containing this file in terminal and running the following command.
            > tensorboard --logdir graph
        """
        tboard_cb = TensorBoard(log_dir='./graph/RNN', histogram_freq=0,
                                write_graph=True, write_images=True)
        early_stopping_cb = EarlyStopping(monitor="val_loss", patience=4)
        return self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs
                       , verbose=verbose, validation_split=0.20
                       , callbacks=[tboard_cb, early_stopping_cb])

class AllCNN(object):
    """
        This class contains a CNN architecture to model Website Traffic
        Fingerprinting using the direction information from undefended data
        """

    def __init__(self, num_features, num_classes):
        """
        :param num_features: number of features (columns) in the data (X)
        :param num_classes: number of unique labels in the data (number of
            websites)
        """
        model = Sequential()
        dense_1 = 256
        l2_lambda = 0.001

        # layer 1
        model.add(Conv1D(filters=32, kernel_size=3
                         , input_shape=(num_features, 1), padding="same"
                         , activation='relu', kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'
                         , kernel_regularizer=l2(l2_lambda)))

        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', strides=2
                         , kernel_regularizer=l2(l2_lambda)))

        # layer 2
        model.add(Conv1D(filters=64, kernel_size=3, padding="same"
                         , activation='relu', kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', strides=2
                         , kernel_regularizer=l2(l2_lambda)))

        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(dense_1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        optimizer = SGD(lr=0.002, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
            'accuracy'])
        self.model = model

    def fit(self, X_train, Y_train, batch_size, epochs, verbose):
        """
        :param X_train: a numpy ndarray of dimension (k x n) containing
            training data
        :param Y_train: a numpy ndarray of dimension (k x 1) containing
            labels for X_train
        :param batch_size: batch size to use for training
        :param epochs: number of epochs for training
        :param verbose: Console print options for training progress.
            0 - silent mode,
            1 - progress bar,
            2 - one line per epoch
        :return: None

        This method start training the model with the given data. The
        training options are configured with tensorboard and early stopping
        callbacks.

        Tensorboard could be launched by navigating to the directory
        containing this file in terminal and running the following command.
            > tensorboard --logdir graph
        """
        tboard_cb = TensorBoard(log_dir='./graph/AllCNN', histogram_freq=0,
                                write_graph=True, write_images=True)
        early_stopping_cb = EarlyStopping(monitor="val_loss", patience=4)
        return self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs
                       , verbose=verbose, validation_split=0.20
                       , callbacks=[tboard_cb, early_stopping_cb])


class CNNModel(object):
    """
    This class contains a CNN architecture to model Website Traffic
    Fingerprinting using the direction information from undefended data
    """
    def __init__(self, num_features, num_classes):
        """
        :param num_features: number of features (columns) in the data (X)
        :param num_classes: number of unique labels in the data (number of
            websites)
        """
        model = Sequential()
        dense_1 = 256
        l2_lambda = 0.001

        # layer 1
        model.add(Conv1D(filters=32, kernel_size=5
                         , input_shape=(num_features, 1), padding="same"
                         , activation='tanh', kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(2))

        # layer 2
        model.add(Conv1D(filters=64, kernel_size=3, activation='tanh'
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(2))

        # layer 3
        model.add(Conv1D(filters=64, kernel_size=3, activation='tanh'
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(2))

        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(dense_1, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
            'accuracy'])
        self.model = model

    def fit(self, X_train, Y_train, batch_size, epochs, verbose):
        """
        :param X_train: a numpy ndarray of dimension (k x n) containing
            training data
        :param Y_train: a numpy ndarray of dimension (k x 1) containing
            labels for X_train
        :param batch_size: batch size to use for training
        :param epochs: number of epochs for training
        :param verbose: Console print options for training progress.
            0 - silent mode,
            1 - progress bar,
            2 - one line per epoch
        :return: None

        This method start training the model with the given data. The
        training options are configured with tensorboard and early stopping
        callbacks.

        Tensorboard could be launched by navigating to the directory
        containing this file in terminal and running the following command.
            > tensorboard --logdir graph
        """
        tboard_cb = TensorBoard(log_dir='./graph/CNN', histogram_freq=0,
                                write_graph=True, write_images=True)
        early_stopping_cb = EarlyStopping(monitor="val_loss", patience=4)
        return self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs
                       , verbose=verbose, validation_split=0.20
                       , callbacks=[tboard_cb, early_stopping_cb])


##################################################################################################################
#
# Below are the DL models used by Srinam et al. Code found at https://github.com/deep-fingerprinting/df
#
##################################################################################################################


# DF model used for WTF-PAD dataset
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.initializers import glorot_uniform

class DFNet_WTFPAD:

    def __init__(self, num_features, classes):
        input_shape = (num_features, 1)
        model = Sequential()
        #Block1
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]

        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],input_shape=input_shape,
                         strides=conv_stride_size[1],padding='same',
                          name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act1'))
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                          name='block1_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding= 'same', name='block1_pool'))
        model.add(Dropout(0.2, name='block1_dropout'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act1'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                          name='block2_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                               padding='same', name='block2_pool'))
        model.add(Dropout(0.2, name='block2_dropout'))

        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                          name='block3_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act1'))
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                          name='block3_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                               padding='same', name='block3_pool'))
        model.add(Dropout(0.2, name='block3_dropout'))

        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                          name='block4_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act1'))
        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                               padding='same', name='block4_pool'))
        model.add(Dropout(0.2, name='block4_dropout'))


        model.add(Flatten(name='flatten'))
        model.add(Dense(512, kernel_initializer = glorot_uniform(seed=0), name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc1_act'))

        model.add(Dropout(0.7, name='fc1_dropout'))

        model.add(Dense(512, kernel_initializer = glorot_uniform(seed=0), name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc2_act'))

        model.add(Dropout(0.5, name='fc2_dropout'))

        model.add(Dense(classes, kernel_initializer = glorot_uniform(seed=0),name='fc3'))
        model.add(Activation('softmax', name="softmax"))

        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
            'accuracy'])
        self.model = model


    def fit(self, X_train, Y_train, batch_size, epochs, verbose):
        """
        :param X_train: a numpy ndarray of dimension (k x n) containing
            training data
        :param Y_train: a numpy ndarray of dimension (k x 1) containing
            labels for X_train
        :param batch_size: batch size to use for training
        :param epochs: number of epochs for training
        :param verbose: Console print options for training progress.
            0 - silent mode,
            1 - progress bar,
            2 - one line per epoch
        :return: None

        This method start training the model with the given data. The
        training options are configured with tensorboard and early stopping
        callbacks.

        Tensorboard could be launched by navigating to the directory
        containing this file in terminal and running the following command.
            > tensorboard --logdir graph
        """
        tboard_cb = TensorBoard(log_dir='./graph/DFNet_WTFPAD', histogram_freq=0,
                                write_graph=True, write_images=True)
        early_stopping_cb = EarlyStopping(monitor="val_loss", patience=4)
        return self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs
                       , verbose=verbose, validation_split=0.20
                       , callbacks=[tboard_cb, early_stopping_cb])



class DFNet_Undefended:
    def __init__(self, num_features, classes):
        input_shape = (num_features, 1)
        model = Sequential()
        #Block1
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]

        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act1'))
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding='same', name='block1_pool'))
        model.add(Dropout(0.1, name='block1_dropout'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act1'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                               padding='same', name='block2_pool'))
        model.add(Dropout(0.1, name='block2_dropout'))

        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act1'))
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                               padding='same', name='block3_pool'))
        model.add(Dropout(0.1, name='block3_dropout'))

        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act1'))
        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                               padding='same', name='block4_pool'))
        model.add(Dropout(0.1, name='block4_dropout'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc1_act'))

        model.add(Dropout(0.7, name='fc1_dropout'))

        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc2_act'))

        model.add(Dropout(0.5, name='fc2_dropout'))

        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc3'))
        model.add(Activation('softmax', name="softmax"))
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
            'accuracy'])
        self.model = model

    def fit(self, X_train, Y_train, batch_size, epochs, verbose):
        """
        :param X_train: a numpy ndarray of dimension (k x n) containing
            training data
        :param Y_train: a numpy ndarray of dimension (k x 1) containing
            labels for X_train
        :param batch_size: batch size to use for training
        :param epochs: number of epochs for training
        :param verbose: Console print options for training progress.
            0 - silent mode,
            1 - progress bar,
            2 - one line per epoch
        :return: None

        This method start training the model with the given data. The
        training options are configured with tensorboard and early stopping
        callbacks.

        Tensorboard could be launched by navigating to the directory
        containing this file in terminal and running the following command.
            > tensorboard --logdir graph
        """
        tboard_cb = TensorBoard(log_dir='./graph/DFNet_Undefended', histogram_freq=0,
                                write_graph=True, write_images=True)
        early_stopping_cb = EarlyStopping(monitor="val_loss", patience=4)
        return self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs
                       , verbose=verbose, validation_split=0.20
                       , callbacks=[tboard_cb, early_stopping_cb])