# -*- coding: utf-8 -*-

import sys; sys.path.insert(0, './mrf_approximator')
from scipy.stats import norm

from acq import *
from plotting import *
from experiments import *
from testconf import experiment_setup

import numpy as np
import os, time, sys


#a complete experiment episode of specified iterations
def test_trial(dim, configurations):
    
    hyperparams = configurations["hyperparams"]
    spaces = configurations["spaces"]
    func = configurations["func"]
    nIter = configurations["learning_iterations"]
    animated = configurations["animated"]
    acq_f = configurations["acq_func"]
    noise = configurations["noise"] # Gaussian variance
    wdir = configurations["working_dir"] # dir for regret evaluation, state snapshots and animation 

    
    np.random.seed(int(time.time() * 1000 % 10000))
    regrets = [experiment(spaces, func, nIter, acq=acq, noise=noise, hyperparams=hyperparams, animated=animated, wdir=wdir) for acq in acq_f]
    if animated:
        for acq in acq_f:
            src_images = os.path.join(configurations["working_dir"], "%dd_%s_*.png"%(dim,acq))
            dst_gif = os.path.join(configurations["working_dir"], "animation_%s.gif"%acq)
            os.system("convert -delay 20 -loop 0 \"%s\" \"%s\""%(src_images, dst_gif)) #imageMacgick is required
            
    return np.array(regrets)
    
    
    
def main(argv):
    if not len(argv) == 2:
        print("Sample Usage: python test.py [dim] [n_trials]")
        return
    dim = int(argv[0])
    n_trials = int(argv[1])
    configurations = experiment_setup(dim)
    regrets_across_trials = [] #2d array that contains regret lists from all experiment trials
    for n in range(n_trials):
        print("test trial %d"%(n+1), "for %d"%n_trials)
        regrets_across_trials.append(test_trial(dim, configurations))
        
    regrets_across_trials = np.array(regrets_across_trials)
    eval_regrets(regrets_across_trials.mean(axis=0), #final evaluation is averaged across trials 
                 [str(acq) for acq in configurations["acq_func"]], 
                 os.path.join(configurations["working_dir"], "eval%dd.eps"%dim), 
                 configurations["xlabel"], 
                 configurations["ylabel"])
    

if __name__ == "__main__":
    main(sys.argv[1: ])
    
    
    
    
    