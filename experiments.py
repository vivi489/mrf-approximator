from mrf_approximator.env import Env
from mrf_approximator.optimizer import Optimizer
from mrf_approximator.click_generator import ClickGenerator
from mrf_approximator.policy import Policy
from mrf_approximator import plotting

from math import log
import numpy as np
import os, sys

def experiment(spaces, func, nIter, wdir, acq=None, noise=0.25, hyperparams=None, animated=None):
    env = Env(spaces, func, noise=noise)
    optimizer = Optimizer(env, hyperparams, edge_normal=True)
    regrets = []
    rsize = 0 # count reward samples; required by some acq
    best = None # count the best reward sample; required by some acq
    _, global_optimum = env.getOptimumIndex() # (best_index, best_value)
    dim = len(spaces)
    os.makedirs(wdir, exist_ok=True)

    for i in range(nIter):
        sys.stdout.write("\033[K")
        print("iter %d"%(i+1), "for %d"%nIter, acq, end='\r', flush=True)
        y, var = optimizer.getPosterior() #start with uniformly random distributions
        
        sample_index = np.argmax(acq.acquire(y, var, best=best, rsize=rsize))\
        if acq is not None else np.random.randint(env.size) #by definition of acquisition functions; randomly samples if not provided

        r = optimizer.sample(sample_index)
        best = r if best is None else max(best, r)
        rsize += 1
        optimizer.updateModel()
        regrets.append(global_optimum - r)
        
        if animated[0] and (i+1)%animated[1]==0: 
            plot_state = [plotting.plot_state_1d, plotting.plot_state_2d, plotting.plot_state_3d][dim - 1]
            state_img_fname_digits = 0; n = nIter
            while n > 0: n //= 10; state_img_fname_digits += 1 #how many digits to name image file sequences
            plot_state(spaces, func, optimizer, os.path.join(wdir, "%dd_%s_iter_%0{}d.png".format(state_img_fname_digits)%(dim, str(acq), i+1)))
    return regrets


def experiment_clicks(spaces, func, nIter, wdir, N=100000, acq=None, hyperparams=None):
    env = Env(spaces, func)
    optimizer = Optimizer(env, hyperparams, edge_normal=True)
    regrets = []
    rsize = 0
    best = None
    _, global_optimum = env.getOptimumIndex()
    #dim = len(spaces)
    os.makedirs(wdir, exist_ok=True)
    
    cgen = ClickGenerator(env, N)
    optimal_clicks = int(N * cgen.getTruth())
    trafficgetter = Policy(acq).trafficGetter() # a func that returns an impression list
    
    for i in range(nIter):
        sys.stdout.write("\033[K")
        print("iter %d"%(i+1), "for %d"%nIter, acq, end='\r', flush=True)
        y, var = optimizer.getPosterior()
        impressions = trafficgetter(N, y, var)
        #print("impressions = ", impressions)
        clicks = []; ctrs = []
        for i in range(env.size):
            click, ctr = cgen.collectAt(impressions[i], i) #ctr is logit(ctr)
            optimizer.sample(i, force_value=ctr) # CTR is transformed from [0: 1] to R
            optimizer.updateModel()
            clicks.append(click) # clicks on all nodes
            ctrs.append(ctr)
            best = ctr if best is None else max(best, ctr)
            rsize += 1
        regrets.append(optimal_clicks - sum(clicks))
    return regrets


        
        
        
        
        