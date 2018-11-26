# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import logit

"""
ClickGenerator makes real-world online click behavior simulation.
Given some environment and total number of ad budget, it generates
clicks and computes CTR at every environment location.
"""
class ClickGenerator:
    def __init__(self, env, N):
        self._env = env
        self._N = N
        self._get_min_max()
        
    def _get_min_max(self):
        _, self.global_max = self._env.get_optimum_index()
        _, self.global_min = self._env.get_optimum_index(reverse=True)
        
    def _min_max_scale(self, val):  # TODO: if global max equals min
        return (val - self.global_min) / (self.global_max - self.global_min)
    
    def get_truth(self):
        return self._min_max_scale(self.global_max)
        
    def collect_at(self, n, index):
        assert n >= 0, "negative impression at the click generator"
        prob = self._env.sample(index, truth=True)
        prob = self._min_max_scale(prob)
        count = np.random.binomial(n, prob) // 4
        ctr = (1 + count) / (2 + n)
        return count, logit(ctr) * 10 + ctr * 50


