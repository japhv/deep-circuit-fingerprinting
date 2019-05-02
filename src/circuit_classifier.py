from data import get_data, split_data
from model import CNN
from keras.utils.np_utils import to_categorical

import numpy as np


def main(config):

    # Load the data and create X and Y matrices
    data = get_data(config)
    num_features = data.shape[1] - 1
    X = data[:, :num_features]
    Y = data[:, -1]

    # split the data into training and test set
    X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.80, balance_dist=True)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # instantiate the CNN model and train on the data
    model = CNN(num_features, Y_train.shape[1])
    model.fit(X_train, Y_train, batch_size=50, epochs=500, verbose=2)

    # Evaluate the trained model on test data and print the accuracy
    score = model.model.evaluate(X_test, Y_test)
    print("\nTest accuracy: ", round(score[1]*100, 2))
    print("Test loss: ", round(score[0], 2))


if __name__ == '__main__':
    # config = {
    #     "raw_data": "../hswf/client/",
    #     "num_websites": 50,
    #     "num_instances": 50,
    # }

    # config = {
    #     "raw_data": "data/defended/client/tamaraw_0501_1207/",
    #     "num_websites": 50,
    #     "num_instances": 50,
    # }

    config = {
        "raw_data": "data/client/",
        "num_websites": 50,
        "num_instances": 10,
    }
    main(config)