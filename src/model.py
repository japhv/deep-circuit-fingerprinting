from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras.regularizers import l2


class CNN(object):
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
        num_filters = [16, 32, 32, 64, 64, 128]
        filter_sizes = [5, 5, 3, 3, 3, 3]
        dense_1 = 256
        l2_lambda = 0.0001

        # layer 1
        model.add(Conv1D(num_filters[0], filter_sizes[0]
                         , input_shape=(num_features, 1), padding="same"
                         , activation='relu', kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(num_filters[1], filter_sizes[1], activation='relu'
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(2))

        # layer 2
        model.add(Conv1D(num_filters[2], filter_sizes[2], activation='relu'
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(num_filters[3], filter_sizes[3], activation='relu'
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(2))

        model.add(Flatten())
        model.add(Dropout(0.7))
        model.add(Dense(dense_1, activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        sgd = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[
            'accuracy'])
        self.model = model
        # print self.model.summary()

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
        tboard_cb = TensorBoard(log_dir='./graph', histogram_freq=0,
                                write_graph=True, write_images=True)
        early_stopping_cb = EarlyStopping(monitor="val_loss", patience=4)
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs
                       , verbose=verbose, validation_split=0.20
                       , callbacks=[tboard_cb, early_stopping_cb])

