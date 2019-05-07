"""
author: Toboure Gambo
author: Japheth Adhavan

This file holds the data reading and processing functions. A lot of code here has been reused from
the Website fingerprinting homework.
"""
import numpy as np
from sklearn.utils import shuffle

def get_data(config: dict) -> np.array:
    """

    :return: a numpy ndarray of dimension (m x (n+1)) containing direction data
        loaded from the files, where `m` is the number of data samples and `n`
        is length of direction packets (restricted to 500 to consume less
        computation time and memory). The last column in the data contains the
        class labels of the `m` samples, which are the website numbers.

    This function loads the data from the files and creates a numpy data matrix
    with each row as a data sample and the columns containing packet direction.
    The last column of the data is the label, which is the website to which the
    instance belongs.
    """

    # modify these parameters in the config file
    data_path = config.get('raw_data')      # data folder
    num_sites = int(config.get('num_websites'))    # 95
    num_instances = int(config.get('num_instances'))   # 100
    file_ext = config.get('file_extension', "")   # No extension
    max_length = 3000    # maximum number of packet directions to use

    # read data from files
    print("loading data...")
    data = []
    for site in range(0, num_sites):
        # print site
        for instance in range(0, num_instances):
            file_name = str(site) + "-" + str(instance)
            # Directory of the raw data
            file_path = data_path + file_name + file_ext
            with open(file_path, "r") as file_pt:
                directions = []
                for line in file_pt:
                    x = line.strip().split('\t')
                    directions.append(1 if float(x[1]) > 0 else -1)
                if len(directions) < max_length:
                    zend = max_length - len(directions)
                    directions.extend([0] * zend)
                elif len(directions) > max_length:
                    directions = directions[:max_length]
                data.append(directions + [site])
    print("done")
    return np.array(data)


def split_data(X, Y, fraction=0.80, balance_dist=False):
    """
    :param X: a numpy ndarray of dimension (m x n) containing data samples
    :param Y: a numpy ndarray of dimension (m x 1) containing labels for X
    :param fraction: a value between 0 and 1, which will be the fraction of
        data split into training and test sets. value of `fraction` will be the
        training data and the rest being test data.
    :param balance_dist: boolean value. The split is performed with ensured
        class balance if the value is true.
    :return: X_train, Y_train, X_test, Y_test

    This function splits the data into training and test datasets.
    """
    X, Y = shuffle(X, Y)
    m, n = X.shape
    split_index = int(round(m*fraction))
    if balance_dist:
        X_train = np.zeros(shape=(split_index, n))
        X_test = np.zeros(shape=(m-split_index, n))
        Y_train = np.zeros(shape=(split_index,))
        Y_test = np.zeros(shape=(m-split_index,))
        labels = np.unique(Y)
        ind1 = 0
        ind2 = 0
        for i in np.arange(labels.size):
            indices = np.where(Y == labels[i])[0]
            split = int(round(len(indices)*fraction))

            X_train[ind1:ind1 + split, :] = X[indices[:split], :]

            X_test[ind2:ind2+(indices.size-split), :] = X[indices[split:], :]

            Y_train[ind1:ind1 + split] = Y[indices[:split]]
            Y_test[ind2:ind2+(indices.size-split)] = Y[indices[split:]]

            ind1 += split
            ind2 += indices.size-split
        X_train, Y_train = shuffle(X_train, Y_train)
        X_test, Y_test = shuffle(X_test, Y_test)
        return X_train, Y_train, X_test, Y_test
    return X[:split_index, :], Y[:split_index], \
        X[split_index:, :], Y[split_index:]
