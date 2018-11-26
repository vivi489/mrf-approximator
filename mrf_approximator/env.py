import operator, functools
from math import sqrt
import numpy as np

"""
IndexManager manipulates index conversion between high-dimensional positions
and flattened offsets

Do not instantiate this class for other usage than for Env
"""
class IndexManager:
    def __init__(self, shape):  # shape is always a dimension array; exception otherwise
        assert len(shape) > 0, "invalid shape definition in index manager"
        self._shape = shape
        self.size = functools.reduce(operator.mul, shape, 1)
 
    def flatten(self, *pos):  # row-major (C-style) order
        assert len(pos) == len(self._shape), "coordinate size does not match shape size"
        prod = 1
        index = 0
        for i in range(len(pos)-1, -1, -1):
            index += pos[i] * prod
            prod *= self._shape[i]
        return index

    def unflatten(self, index):  # return coordinates in C-style order
        indices = []
        prod = self.size
        for i in range(len(self._shape)):
            prod //= self._shape[i]
            indices.append(int(index // prod))
            index %= prod
        # assert index == 0, "reduced index is not zero"
        return indices  # a list

    def isadj(self, idx1, idx2):
        pos1 = np.array(self.unflatten(idx1))
        pos2 = np.array(self.unflatten(idx2))
        mask = pos1 != pos2
        # adjacent iff two coordinates differs at 1 index by 1 offset
        return len(pos1[mask]) == 1 and abs(pos1[mask][0]-pos2[mask][0]) == 1


"""
Env handles mapping between grid axis values and space coordinates;
acquires sample/true rewards; describes global min/max property
"""
class Env:
    def __init__(self, xn, func, noise=0.25):
        self._xn = xn  # grid axes
        self.func = func  # environment
        self._noise = noise  # Gaussian variance
        self._idxMgr = IndexManager([len(x) for x in xn])
        self.size = self._idxMgr.size  # size of the discrete space
        self.dim = len(xn)  # discrete space dimension

    # coordinates to index (axis positions to values on axes)
    # transform a location into a feature
    def _pos2x(self, *pos): 
        # the last grid axis in self._xn is indexed by the last (innermost) coordinate
        return [self._xn[i][pos[i]] for i in range(len(self._xn))]

    def _fpos(self, *pos):
        x = self._pos2x(*pos)
        if len(x) == 1: x = x[0]  # ripped off from an array in case of 1d indices
        return self.func(x)

    def _fi(self, index):
        pos = self._idxMgr.unflatten(index)
        return self._fpos(*pos)

    def sample(self, index, truth=False):
        r = self._fi(index) 
        if not truth:
            r += np.random.normal(0, sqrt(self._noise))
        return r
    
    # returns a self.size * self.size matrix with A[j, i] = 1 for adjacent indices
    def get_adj(self):
        matrix_adj = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self._idxMgr.isadj(i, j):
                    matrix_adj[i, j] = 1
                    matrix_adj[j, i] = 1
        return matrix_adj

    # return minimum value and index for reverse==True
    def get_optimum_index(self, reverse=False):
        val, index = None, -1
        for i in range(self.size):
            if not reverse:
                if index < 0 or self._fi(i) > val:
                    index = i
                    val = self._fi(i)
            else:
                if index < 0 or self._fi(i) < val:
                    index = i
                    val = self._fi(i)
        return index, val
