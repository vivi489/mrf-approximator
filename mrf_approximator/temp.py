# -*- coding: utf-8 -*-

from env import Env
from optimizer import Optimizer
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from acq import *

import matplotlib.pyplot as plt
import numpy as np




spaces = [np.linspace(-5, 5, 30), np.linspace(-4, 4, 20)]



mean1 = [-3, 2]
cov1 = [[0.5, 0], [0, 0.5]]

mean2 = [0, 0]
cov2 = [[2.25, 0], [0, 2.25]]

mean3 = [1, -2]
cov3 = [[1.25, 0], [0, 1.25]]
  
func = lambda x: (multivariate_normal.pdf(x, mean=mean1, cov=cov1) + 
                  multivariate_normal.pdf(x, mean=mean2, cov=cov2) + 
                  multivariate_normal.pdf(x, mean=mean3, cov=cov3)) * 100

env = Env(spaces, func, noise=0.1)
optimizer = Optimizer(env, edge_normal=True)

acquisition_params = {
    "par": 0.01,
    "dim": 1,
    "eps": 0.2,
    "delta": 0.9
}
acq = TS(acquisition_params)
rsize = 0
best = None
regrets = []

ii, global_optimum = env.getOptimumIndex()
#for _ in range(1):
#    ir = optimizer.sample(ii)
#    optimizer.updateModel()
print("best reward", global_optimum, "best index=", ii)
#y, var = optimizer.getPosterior()
#print("y", y, np.argmax(y))


for i in range(50):
    y, var = optimizer.getPosterior()
    sample_index = np.argmax(acq.acquire(y, var, best=best, rsize=rsize))
    print("sample_index=", sample_index)
    r = optimizer.sample(sample_index)
    best = r if best is None else max(best, r)
    rsize += 1
    optimizer.updateModel()
    regrets.append(global_optimum - r)
    
    
print(regrets)