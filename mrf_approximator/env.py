import operator, functools
from math import sqrt
import numpy as np


class IndexManager:
    def __init__(self, shape):
        assert len(shape) > 0, "invalid shape definition in index manager"
        self._shape = shape
        self.size = functools.reduce(operator.mul, shape, 1)
 
    def flatten(self, *pos): # row-major (C-style) order
        assert len(pos) == len(self._shape), "coordinate size does not match shape size"
        prod = 1
        index = 0
        for i in range(len(pos)-1, -1, -1):
            index += pos[i] * prod
            prod *= self._shape[i]
        return index

    def unflatten(self, index):
        indices = []
        prod = self.size
        for i in range(len(self._shape)):
            prod //= self._shape[i]
            indices.append(int(index // prod))
            index %= prod
        #assert index == 0, "reduced index is not zero"
        return indices

    def isAdj(self, idx1, idx2):
        pos1 = np.array(self.unflatten(idx1))
        pos2 = np.array(self.unflatten(idx2))
        mask = pos1!=pos2
        return len(pos1[mask])==1 and abs(pos1[mask][0]-pos2[mask][0])==1

class Env:
    def __init__(self, xn, func, noise=0.25):
        self._xn = xn
        self.func = func
        self._noise =noise
        self._idxMgr = IndexManager([len(x) for x in xn])
        self.size = self._idxMgr.size
        self.func = func
        self.dim = len(xn)

    def _pos2x(self, *pos): #index to independent variables
        return [self._xn[i][pos[i]] for i in range(len(self._xn))]

    def _fpos(self, *pos):
        x = self._pos2x(*pos)
        if len(x) == 1: x = x[0] #rip off an array in case of 1d indices
        return self.func(x)

    def _fi(self, index):
        pos = self._idxMgr.unflatten(index)
        return self._fpos(*pos)

    def sample(self, index, truth=False):
        r = self._fi(index) 
        if not truth: r += np.random.normal(0, sqrt(self._noise))
        return r
    
    def getAdj(self):
        matrixAdj = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self._idxMgr.isAdj(i, j):
                    matrixAdj[i, j] = 1
                    matrixAdj[j, i] = 1
        return matrixAdj

    def getOptimumIndex(self, reverse=False): #minimum for reverse==True
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
