"""
author: Toboure Gambo
author: Japheth Adhavan

This file holds visualization functions
"""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_accuracy(title, results):
    """
    Plots a list of accuracies for a given model
    :param title: The title of the model for which the accuracy is plotted
    :param results: Dictionary of the model accuracies
    :return:
    """
    fig, ax = plt.subplots()

    os.makedirs("./plots", exist_ok=True)

    ax.set_title("{} based model".format(title))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")

    # Adjust x-axis ticks
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    colors  = ["red", "blue", "purple"]
    markers = [".", "o", "x"]

    for i , (dataset, acc) in enumerate(results.items()):
        epochs = len(acc)
        x = np.linspace(1, epochs, epochs)
        ax.plot(x, acc, color=colors[i], label=dataset, marker=markers[i])

    ax.legend()

    fig.savefig("./plots/{}".format(title))
