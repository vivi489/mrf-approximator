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
    
    spaces = [np.linspace(-10, 10, 5), np.linspace(-6, 6, 6), np.linspace(-5, 5, 10)]
        
    mean1 = [8, 4, 2]
    cov1 = np.eye(3) * 5
    
    mean2 = [-8, -4, -2]
    cov2 = np.eye(3) * 2.5
    
    mean3 = [-0.5, -1, 0.5]
    cov3 = np.eye(3) * 4.0
    
    func = lambda x: (multivariate_normal.pdf(x, mean=mean1, cov=cov1) + 
         multivariate_normal.pdf(x, mean=mean2, cov=cov2) + 
         multivariate_normal.pdf(x, mean=mean3, cov=cov3)) * 1000

    nIter = 1000
    animated = (False, 100)
    acq_f = [EI(acquisition_params), PI(acquisition_params), UCB(acquisition_params), TS(acquisition_params), EPS(acquisition_params)]
    #acq_f = [TS(acquisition_params)]
    regrets = [experiment(spaces, func, nIter, acq=acq, hyperparams=hyperparams, animated=animated) for acq in acq_f]
    if animated:
        for acq in acq_f:
            os.system("convert -delay 20 -loop 0 ./test3d/3d_%s_*.png ./test3d/rewards_%s.gif"%(acq, acq))
    eval_regrets(regrets, [str(acq) for acq in acq_f], "./test3d/eval3d.eps", "iteration", "average regret")

if __name__ == "__main__":
    main()

