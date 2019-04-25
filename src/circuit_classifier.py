from data import get_data, split_data
from model import CNN
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier

import numpy as np


def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = layers.average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns


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
    cnn_model = CNN(num_features, Y_train.shape[1])
    rfc_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

    ensemble_model = ensembleModels([cnn_model, rfc_model], [X_train_cnn, X_train_rfc])

    # Evaluate the trained model on test data and print the accuracy
    score = ensemble_model.evaluate(X_test, Y_test, batch_size=100)
    print("\nTest accuracy: ", round(score[1]*100, 2))
    print("Test loss: ", round(score[0], 2))


if __name__ == '__main__':
    config = {
        "raw_data": "../hswf/client/",
        "num_websites": 50,
        "num_instances": 50,
    }
    main(config)