# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import logit

class ClickGenerator:
    def __init__(self, env, N):
        self._env = env
        self._N = N
        self._getMinMax()
        
    def _getMinMax(self):
        _, self.global_max = self._env.getOptimumIndex()
        _, self.global_min = self._env.getOptimumIndex(reverse=True)
        #print(self.global_min, self.global_max)
        
    def _minMaxScale(self, val): #TODO: if global max equals min
        return (val - self.global_min) / (self.global_max - self.global_min)
    
    def getTruth(self):
        return self._minMaxScale(self.global_max)
        
    def collectAt(self, n, index):
        assert n>=0, "negative impression at the click generator"
        prob = self._env.sample(index, truth=True)
        prob = self._minMaxScale(prob)
        count = np.random.binomial(n, prob)
        ctr = (1 + count) / (2 + n)
        return count, logit(ctr)
    
