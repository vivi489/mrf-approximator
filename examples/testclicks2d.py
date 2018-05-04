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
    
    spaces = [np.linspace(-5, 5, 20), np.linspace(-4, 4, 15)]
        
    mean1 = [-3, 3]
    cov1 = [[0.25, 0], [0, 0.25]]
    
    mean2 = [0, 0]
    cov2 = [[1.25, 0], [0, 1.25]]
    
    mean3 = [2, -2]
    cov3 = [[0.75, 0], [0, 0.75]]
      
    func = lambda x: (multivariate_normal.pdf(x, mean=mean1, cov=cov1) + 
                      multivariate_normal.pdf(x, mean=mean2, cov=cov2) + 
                      multivariate_normal.pdf(x, mean=mean3, cov=cov3)) * 100

    nIter = 100
    #acq_f = [EI(acquisition_params), PI(acquisition_params), UCB(acquisition_params), TS(acquisition_params), EPS(acquisition_params)]
    acq_f = [UCB(acquisition_params)]
    #acq_f = [UCB(acquisition_params), TS(acquisition_params)]
    regrets = [experiment_clicks(spaces, func, nIter, N=100000, acq=acq, hyperparams=hyperparams) for acq in acq_f]
    eval_regrets(regrets, [str(acq) for acq in acq_f], "./testclick2d/eval2dclicks.eps", "Iteration", "Average Click Loss")
    
if __name__ == "__main__":
    main()
