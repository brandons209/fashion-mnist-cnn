import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_history(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_graph(x, y, xticks, yticks, x_label, y_label, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.plot(x, y, linewidth=2, marker='o', markersize=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(True)
    plt.show()

def plot_metrics(history_data):
    epochs = np.arange(len(history_data['loss'])) + 1
    for key in history_data:
        if 'loss' in key:
            plot_graph(epochs, history_data[key], epochs, np.linspace(0, 0.7, num=15), "Epochs", key, key+" per Epoch")
        elif 'acc' in key:
            plot_graph(epochs, history_data[key], epochs, np.linspace(0, 1, num=15), "Epochs", key, key+"per Epoch")

plot_metrics(load_history(sys.argv[1]))
