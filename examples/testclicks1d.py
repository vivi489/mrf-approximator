# -*- coding: utf-8 -*-

import sys; sys.path.insert(0, './mrf_approximator')
from scipy.stats import norm

from acq import *
from plotting import *
from experiments import *

import numpy as np
import os


def main(*argv):
    hyperparams=None
    acquisition_params = {
        "par": 0.01,
        "dim": 1,
        "eps": 0.2,
        "delta": 0.9
    }
    
    space = np.linspace(-10, 10, 200)
    func = lambda x: (norm.pdf(x, loc=2, scale=0.7) + 
                      norm.pdf(x, loc=-5, scale=1.0) + 
                      norm.pdf(x, loc=5, scale=0.5)) * 100
    nIter = 100
    acq_f = [EI(acquisition_params), PI(acquisition_params), UCB(acquisition_params), TS(acquisition_params), EPS(acquisition_params)]
    #acq_f = [UCB(acquisition_params)]
    #acq_f = [UCB(acquisition_params), TS(acquisition_params)]
    regrets = [experiment_clicks([space], func, nIter, N=100000, acq=acq, hyperparams=hyperparams) for acq in acq_f]
    eval_regrets(regrets, [str(acq) for acq in acq_f], "./testclick1d/eval1dclicks.eps", "Iteration", "Average Click Loss")
    
if __name__ == "__main__":
    main()
