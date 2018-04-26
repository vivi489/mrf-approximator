import numpy as np
from node import Node
from scipy import linalg

class Optimizer:
    def __init__(self, env, edge_normal=False, **hyperparams):
        self.hyperparams = {}
        self.hyperparams["alpha"] = 0.001 if "alpha" not in hyperparams else hyperparams["alpha"]
        self.hyperparams["gamma_y"] = 0.01 if "gamma_y" not in hyperparams else hyperparams["gamma_y"]
        self.hyperparams["gamma"] = 0.02 * env.dim if "gamma" not in hyperparams else hyperparams["gamma"]
        self.hyperparams["gamma_0"] = 0.01 * self.hyperparams["gamma"] if "gamma_0" not in hyperparams else hyperparams["gamma_0"]
        self.edge_normal = edge_normal
        self.env = env
        self.nodes = [Node() for _ in range(self.env.size)]
        self.matrixA, self.matrixB = self._getAB()
        
    def _getAB(self):
        matrixA = self.env.getAdj()
        neighbor_count = None
        if self.edge_normal:
            neighbor_count = np.array([self.env.dim * 2] * self.env.size)
        else:
            neighbor_count = np.array([matrixA[i, :].sum() for i in range(self.env.size)])
            
        matrixA *= -self.hyperparams["gamma_y"]
        matrixB = np.diag([n.count * self.hyperparams["gamma"] + self.hyperparams["gamma_0"] for n in self.nodes])
        for i in range(self.env.size):
            matrixA[i, i] = self.hyperparams["gamma_y"] * neighbor_count[i] + matrixB[i, i]
        
        return matrixA, matrixB
        
    def _updateAB(self):
        for i in range(self.env.size):
            self.matrixA[i, i] -= self.matrixB[i, i]
        self.matrixB = np.diag([n.count * self.hyperparams["gamma"] + self.hyperparams["gamma_0"] for n in self.nodes])
        for i in range(self.env.size):
            self.matrixA[i, i] += self.matrixB[i, i]
            
    def _getMuTilde(self, n):
        mu_tilde = self.hyperparams["gamma"]* n.mean * n.count + self.hyperparams["gamma_0"] * self.hyperparams["alpha"]
        mu_tilde /= n.count * self.hyperparams["gamma"] + self.hyperparams["gamma_0"]
        return mu_tilde

    def updateModel(self):
        self._updateAB()
    
    def sample(self, index):
        r = self.env.sample(index)
        self.nodes[index].update(r)
        return r

    def getPosterior(self):
        mu_tilde = np.array([self._getMuTilde(n) for n in self.nodes])
        An = linalg.inv(self.matrixA)
        #print(An.shape, self.matrixB.shape, mu_tilde[:, np.newaxis].shape)
        y = (An.dot(self.matrixB).dot(mu_tilde[:, np.newaxis])).flatten()
        var = np.diag(An)
        return y, var
    