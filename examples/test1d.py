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
    
    space = np.linspace(-10, 10, 1000)
    func = lambda x: (norm.pdf(x, loc=1, scale=0.7) + 
                  norm.pdf(x, loc=-5, scale=1.0) + 
                  norm.pdf(x, loc=5, scale=0.5)) * 100
    nIter = 200
    animated = (True, 40)
    acq_f = [EI(acquisition_params), PI(acquisition_params), UCB(acquisition_params), TS(acquisition_params), EPS(acquisition_params)]
    #acq_f = [EI(acquisition_params), PI(acquisition_params), UCB(acquisition_params), EPS(acquisition_params)]
    #acq_f = [TS(acquisition_params)]
    regrets = [experiment([space], func, nIter, wdir="./test1d", acq=acq, hyperparams=hyperparams, animated=animated) for acq in acq_f]
    if animated:
        for acq in acq_f:
            os.system("convert -delay 20 -loop 0 ./test1d/1d_%s_*.png ./test1d/rewards_%s.gif"%(acq, acq))
    eval_regrets(regrets, [str(acq) for acq in acq_f], "./test1d/eval1d.eps", "iteration", "average regret")
    
if __name__ == "__main__":
    main()
