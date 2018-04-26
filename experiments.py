# -*- coding: utf-8 -*-

import sys; sys.path.insert(0, './mrf_approximator')
from env import Env
from optimizer import Optimizer
#from scipy.stats import norm
from acq import *
from plotting import *

import numpy as np
import os

def experiment(spaces, func, nIter, acq=None, noise=0.25 ,hyperparams=None, animated=False):
    env = Env(spaces, func, noise=noise)
    optimizer = Optimizer(env, edge_normal=True)
    regrets = []
    rsize = 0
    best = None
    _, global_optimum = env.getOptimumIndex()
    dim = len(spaces)
    os.makedirs("./test%dd"%dim, exist_ok=True)

    for i in range(nIter):
        y, var = optimizer.getPosterior()
        
        sample_index = np.argmax(acq.acquire(y, var, best=best, rsize=rsize))\
        if acq is not None else np.random.randint(env.size)
#        print("sample_index=", sample_index)
        r = optimizer.sample(sample_index)
        best = r if best is None else max(best, r)
        rsize += 1
        optimizer.updateModel()
        regrets.append(global_optimum - r)
        
        if animated:
            plot_state = [plot_state_1d, plot_state_2d, plot_state_3d][dim - 1]
            plot_state(spaces, func, optimizer, "./test%dd/%dd_%s_iter_%d.png"%(dim, dim, acq.name, i))
            
    if animated:
        os.system("convert -delay 20 -loop 0 ./test%dd/%dd_%s_*.png ./test%dd/rewards.gif"%(dim, dim, acq.name, dim))
    return regrets
