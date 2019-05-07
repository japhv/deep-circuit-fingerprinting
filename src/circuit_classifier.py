"""
author: Toboure Gambo
author: Japheth Adhavan

This is the main classifier file. A lot of code here has been reused from
the Website fingerprinting homework.
"""
import keras
from data import get_data, split_data
from model import CNNModel, RNNModel, DFNet_Undefended, DFNet_WTFPAD, AllCNN
from keras.utils.np_utils import to_categorical
from visualize import plot_accuracy

#######################################################################################################################
# Seed values so that we get reproducible results
# Apparently you may use different seed values at each stage
seed_value= 503

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)
#######################################################################################################################

def train(config, Model):

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
    model = Model(num_features, Y_train.shape[1])
    history = model.fit(X_train, Y_train, batch_size=128, epochs=100, verbose=2)

    # Evaluate the trained model on test data and print the accuracy
    score = model.model.evaluate(X_test, Y_test)
    print("\nTest accuracy: ", round(score[1]*100, 2))
    print("Test loss: ", round(score[0], 2))

    return history

#######################################################################################################################

def plot_model_accuracy(Model, model_name):
    config = {
        "num_websites": 50,
        "num_instances": 10,
    }

    results = {}

    config["raw_data"] = "./data/undefended/client/"
    results["Undefended"] = train(config, Model).history["acc"]
    keras.backend.clear_session()

    config["raw_data"] = "./data/defended/client/wtf-pad/"
    results["WTF-PAD"] = train(config, Model).history["acc"]
    keras.backend.clear_session()
    config["raw_data"] = "./data/defended/client/tamaraw/"
    results["TAMARAW"] = train(config, Model).history["acc"]
    keras.backend.clear_session()

    plot_accuracy(model_name, results)

#######################################################################################################################

if __name__ == '__main__':
    models_to_train = [(CNNModel, "CNN"), (RNNModel, "RNN"), (AllCNN, "All-CNN")]

    for Model, name in models_to_train:
        plot_model_accuracy(Model, name)

#######################################################################################################################
