# -*- coding: utf-8 -*-

import sys; sys.path.insert(0, './mrf_approximator')
from scipy.stats import multivariate_normal

from acq import *
from plotting import *
from experiments import *

import numpy as np


def main(*argv):
    hyperparams=None
    acquisition_params = {
        "par": 0.01,
        "dim": 1,
        "eps": 0.2,
        "delta": 0.9
    }
    
    spaces = [np.linspace(-5, 5, 30), np.linspace(-4, 4, 20)]
        
    mean1 = [-3, 3]
    cov1 = [[0.25, 0], [0, 0.25]]
    
    mean2 = [0, 0]
    cov2 = [[1.25, 0], [0, 1.25]]
    
    mean3 = [2, -2]
    cov3 = [[0.75, 0], [0, 0.75]]
      
    func = lambda x: (multivariate_normal.pdf(x, mean=mean1, cov=cov1) + 
                      multivariate_normal.pdf(x, mean=mean2, cov=cov2) + 
                      multivariate_normal.pdf(x, mean=mean3, cov=cov3)) * 100
        
    nIter = 1000
    animated = (True, 100)
    acq_f = [EI(acquisition_params), PI(acquisition_params), UCB(acquisition_params), TS(acquisition_params), EPS(acquisition_params)]
    #acq_f = [TS(acquisition_params)]
    regrets = [experiment(spaces, func, nIter, acq=acq, hyperparams=hyperparams, animated=animated) for acq in acq_f]
    if animated:
        for acq in acq_f:
            os.system("convert -delay 20 -loop 0 ./test2d/2d_%s_*.png ./test2d/rewards_%s.gif"%(acq, acq))
    eval_regrets(regrets, [str(acq) for acq in acq_f], "./test2d/eval2d.eps", "iteration", "average regret")

if __name__ == "__main__":
    main()
