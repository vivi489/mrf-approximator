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
    
    space = np.linspace(-10, 10, 500)
    func = lambda x: (norm.pdf(x, loc=0, scale=0.7) + 
                      norm.pdf(x, loc=-5, scale=1.0) + 
                      norm.pdf(x, loc=5, scale=0.5)) * 100
    nIter = 50
    #acq_f = [EI(acquisition_params), PI(acquisition_params), UCB(acquisition_params), TS(acquisition_params), EPS(acquisition_params)]
    acq_f = [TS(acquisition_params)]
    regrets = [experiment([space], func, nIter, acq=acq, hyperparams=hyperparams, animated=True) for acq in acq_f]
    os.system("convert -delay 20 -loop 0 ./temp/1d*.png ./test1d/rewards.gif", shell=True)
    eval_regrets(regrets, [acq.name for acq in acq_f], "./test1d/eval1d.png")
    
if __name__ == "__main__":
    main()
