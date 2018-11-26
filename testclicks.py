# -*- coding: utf-8 -*-

import sys
from scipy.stats import norm

from mrf_approximator.acq import *
from mrf_approximator.plotting import *
from experiments import *
from testconf import experiment_setup_testclicks

import numpy as np
import os, time, sys


# a complete experiment episode of specified iterations
def test_trial(configurations):
    
    hyperparams = configurations["hyperparams"]
    spaces = configurations["spaces"]
    func = configurations["func"]
    nIter = configurations["learning_iterations"]
    acq_f = configurations["acq_func"]
    wdir = configurations["working_dir"]  # dir for regret evaluation and temp resources
    N = configurations["N"]  # number of total clicks
    
    np.random.seed(int(time.time() * 1000 % 10000))
    regrets = [experiment_clicks(spaces, func, nIter, wdir, N=N, acq=acq, hyperparams=hyperparams) for acq in acq_f]
    
    return np.array(regrets)
    

def main(argv):
    if not len(argv) == 2:
        print("Sample Usage: python testclicks.py [dim] [n_trials]")
        return
    dim = int(argv[0])
    n_trials = int(argv[1])
    configurations = experiment_setup_testclicks(dim)
    regrets_across_trials = []  # 2d array that contains regret lists from all experiment trials
    for n in range(n_trials):
        print("test trial %d"%(n+1), "for %d"%n_trials)
        regrets_across_trials.append(test_trial(configurations))
        
    regrets_across_trials = np.array(regrets_across_trials)
    eval_regrets(regrets_across_trials.mean(axis=0),  # final evaluation is averaged across trials
                 [str(acq) for acq in configurations["acq_func"]], 
                 os.path.join(configurations["working_dir"], "eval%dd.eps"%dim), 
                 configurations["xlabel"],
                 configurations["ylabel"],
                 ylog=True)
    

if __name__ == "__main__":
    main(sys.argv[1:])
