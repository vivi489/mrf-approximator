import numpy as np
from scipy import linalg
from .node import Node

"""
the core driving class for random markov field optimization
"""
class Optimizer:
    def __init__(self, env, hyperparams, edge_normal=False):
        self.hyperparams = dict()
        self.hyperparams["alpha"] = 0.001 if hyperparams is None else hyperparams["alpha"]
        self.hyperparams["gamma_y"] = 0.01 if hyperparams is None else hyperparams["gamma_y"]
        self.hyperparams["gamma"] = 0.02 * env.dim if hyperparams is None else hyperparams["gamma"]
        self.hyperparams["gamma_0"] = 0.01 * self.hyperparams["gamma"] if hyperparams is None else hyperparams["gamma_0"]
        self.edge_normal = edge_normal  # edge normalization
        self.env = env
        self.nodes = [Node() for _ in range(self.env.size)]
        self.matrixA, self.matrixB_diag = self._getAB()
        
    def _getAB(self):  # return an adjacent matrix and a vector representing diagonal
        matrixA = self.env.get_adj()
        neighbor_count = None
        if self.edge_normal:
            neighbor_count = np.array([self.env.dim * 2] * self.env.size)
        else:
            neighbor_count = np.array([matrixA[i, :].sum() for i in range(self.env.size)])
            
        matrixA *= -self.hyperparams["gamma_y"]
        matrixB_diag = np.array([n.count * self.hyperparams["gamma"] + self.hyperparams["gamma_0"] for n in self.nodes])
        for i in range(self.env.size):
            matrixA[i, i] = self.hyperparams["gamma_y"] * neighbor_count[i] + matrixB_diag[i]

        return matrixA, matrixB_diag
        
    def _updateAB(self):  # update model posterior when nodes get altered
        for i in range(self.env.size):
            self.matrixA[i, i] -= self.matrixB_diag[i]
        self.matrixB_diag = np.array([n.count * self.hyperparams["gamma"] + self.hyperparams["gamma_0"] for n in self.nodes])
        for i in range(self.env.size):
            self.matrixA[i, i] += self.matrixB_diag[i]

    def _get_mu_tilde(self, n):
        mu_tilde = self.hyperparams["gamma"]* n.mean * n.count + self.hyperparams["gamma_0"] * self.hyperparams["alpha"]
        mu_tilde /= n.count * self.hyperparams["gamma"] + self.hyperparams["gamma_0"]
        return np.array(mu_tilde)

    def update_model(self):
        self._updateAB()
    
    def sample(self, index, force_value=None):  # use force_value to update node stats without sampling from env
        r = self.env.sample(index) if force_value is None else force_value
        self.nodes[index].update(r)
        return r

    def get_posterior(self):
        mu_tilde = np.array([self._get_mu_tilde(n) for n in self.nodes])
        An = linalg.inv(self.matrixA)
        y = An.dot((self.matrixB_diag * mu_tilde)[:, np.newaxis]).flatten()
        var = np.diag(An)
        return y, var

    def print_nodes(self):
        for n in self.nodes:
            print(str(n))

