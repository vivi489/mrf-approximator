import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

color_map = defaultdict(lambda: "grey")
color_map.update({"ucb": "black", "pi": "purple", "ei": "blue", "ts": "red", "eps": "green"})
# marker_map = {"ucb": "|", "pi": ".", "ei": "+", "ts": "x", "eps": "_"}

def eval_regrets(regrets, labels, path, xlabel, ylabel, ylog=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylog:
        plt.yscale("log")
    
    handles = []
    for regret, acq in zip(regrets, labels):
        average = []
        cur = 0
        for r in regret:
            N = len(average) + 1
            cur = cur * (N - 1) / N + r / N
            average.append(cur)
        handle, = ax.plot(list(range(len(regret))), average, color=color_map[acq])
        handles.append(handle)

    ax.legend(handles, [l.upper() for l in labels])
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_state_1d(spaces, func, optimizer, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    space = spaces[0]
    handles = []
    handle, = ax.plot(space, func(space), color="black")
    handles.append(handle)
    mu, _ = optimizer.get_posterior()
    handle, = ax.plot(space, mu, color="red")
    handles.append(handle)
    ax.legend(handles, ["truth", "inference"])    
    plt.savefig(path)
    plt.close()


def plot_state_2d(spaces, func, optimizer, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    Y, X = np.meshgrid(*spaces[::-1])
    
    F = np.empty(X.shape + (2,))
    F[:, :, 0] = X; F[:, :, 1] = Y
    
    ax.plot_wireframe(X, Y, func(F), color="black", label="truth")
    
    y, _ = optimizer.get_posterior()
    Z = np.empty(X.shape)  # to store current inference mu
    
    y, _ = optimizer.get_posterior()
    for j in range(Z.shape[0]):
        for i in range(Z.shape[1]):
            Z[j, i] = y[j * Z.shape[1] + i]
    
    ax.plot_wireframe(X, Y, Z, color="red", label="inference")
    ax.legend()
    plt.savefig(path)
    plt.close()


def plot_state_3d(spaces, func, optimizer, path):
    fig = plt.figure(figsize=(32, 10))
    ax = fig.add_subplot(121, projection='3d')
    
    Z, Y, X = np.meshgrid(*spaces[::-1])
    
    F = np.empty(X.shape + (3,))
    F[:, :, :, 0] = X; F[:, :, :, 1] = Y; F[:, :, :, 2] = Z
    
    path3DCollection = ax.scatter(X, Y, Z, c=func(F).flatten(), cmap=cm.coolwarm)
    ax.set_title("truth", size=20)
    fig.colorbar(path3DCollection, shrink=0.7, aspect=20)
    
    y, _ = optimizer.get_posterior()
    W = np.empty(X.shape)  # to store current inference mu
    y, _ = optimizer.get_posterior()
    for k in range(W.shape[0]):
        for j in range(W.shape[1]):
            for i in range(W.shape[2]):
                W[k, j, i] = y[k * W.shape[1] * W.shape[2] + j * W.shape[2] + i]
    
    ax = fig.add_subplot(122, projection='3d')
    path3DCollection = ax.scatter(X, Y, Z, c=W.flatten(), cmap=cm.coolwarm)
    ax.set_title("inference", size=20)
    fig.colorbar(path3DCollection, shrink=0.7, aspect=20)
    
    plt.savefig(path)
    plt.close()