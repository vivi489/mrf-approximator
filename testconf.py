import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm

from mrf_approximator import acq
import os


def _get_spaces(dim):
    return {
        1: [np.linspace(-10, 10, 1000)],
        2: [np.linspace(-5, 5, 30), np.linspace(-4, 4, 20)],
        3: [np.linspace(-8, 8, 16), np.linspace(-6, 6, 10), np.linspace(-5, 5, 8)]
    }[dim]


def _get_spaces_clicks(dim):
    return {
        1: [np.linspace(-5, 5, 100)],
        2: [np.linspace(-5, 5, 20), np.linspace(-4, 4, 15)],
        3: [np.linspace(-10, 10, 5), np.linspace(-6, 6, 6), np.linspace(-5, 5, 10)]
    }[dim]


def _get_func(dim):
    
    def func1d():
        return lambda x: (norm.pdf(x, loc=-1, scale=1.5) +
                          norm.pdf(x, loc=-3, scale=0.15) +
                          norm.pdf(x, loc=3, scale=0.7))
                          
    def func2d():
        mean1 = [-3, 3]
        cov1 = [[0.25, 0], [0, 0.25]]
        
        mean2 = [0, 0]
        cov2 = [[1.25, 0], [0, 1.25]]
        
        mean3 = [2, -2]
        cov3 = [[0.75, 0], [0, 0.75]]
        return lambda x: (multivariate_normal.pdf(x, mean=mean1, cov=cov1) +
                          multivariate_normal.pdf(x, mean=mean2, cov=cov2) +
                          multivariate_normal.pdf(x, mean=mean3, cov=cov3)) * 10
        
    def func3d():
        mean1 = [6, 4, 2]
        cov1 = np.eye(3) * 5
        
        mean2 = [-6, -4, -2]
        cov2 = np.eye(3) * 2.5
        
        mean3 = [-0.5, -1, 0.5]
        cov3 = np.eye(3) * 4.0
    
        return lambda x: (multivariate_normal.pdf(x, mean=mean1, cov=cov1) +
                          multivariate_normal.pdf(x, mean=mean2, cov=cov2) +
                          multivariate_normal.pdf(x, mean=mean3, cov=cov3)) * 1000

    return [func1d, func2d, func3d][dim-1]()


def experiment_setup(dim):
    hyperparams = {
        "alpha": 0.001,
        "gamma_y": 0.01,
        "gamma": 0.02 * dim,
        "gamma_0": 0.01 * (0.02 * dim)
    }
            
    acquisition_params = {
        "par": 0.01,
        "dim": 1,
        "eps": 0.2,
        "delta": 0.9
    }

    return {
        "hyperparams": hyperparams,
        "acquisition_params": acquisition_params,
        "noise": 0.25,
        "spaces": [
            _get_spaces(1),
            _get_spaces(2),
            _get_spaces(3)
        ][dim-1],
        "func": [
            _get_func(1),
            _get_func(2),
            _get_func(3)
        ][dim-1],
        "learning_iterations": 1000,
        "animated": (False, 40),
        "acq_func": [
            acq.EI(acquisition_params),
            acq.PI(acquisition_params),
            acq.UCB(acquisition_params),  
            acq.EPS(acquisition_params),
            acq.TS(acquisition_params)
        ],
        "xlabel": "Iteration",
        "ylabel": "Average Regret",
        "working_dir": [
            os.path.join(os.getcwd(), "output", "test1d"),
            os.path.join(os.getcwd(), "output", "test2d"),
            os.path.join(os.getcwd(), "output", "test3d")
        ][dim-1]
    }
    

def experiment_setup_testclicks(dim):
    hyperparams = {
        "alpha": 0.001,
        "gamma_y": 0.01,
        "gamma": 0.02 * dim,
        "gamma_0": 0.01 * (0.02 * dim)
    }
            
    acquisition_params = {
        "par": 0.01,
        "dim": dim,
        "eps": 0.2,
        "delta": 0.9
    }

    return {
        "hyperparams": hyperparams,
        "acquisition_params": acquisition_params,

        "spaces": [
            _get_spaces_clicks(1),
            _get_spaces_clicks(2),
            _get_spaces_clicks(3)
        ][dim-1],
        "func": [
            _get_func(1),
            _get_func(2),
            _get_func(3)
        ][dim-1],
        "learning_iterations": 50,
        "acq_func": [
            acq.EI(acquisition_params),
            acq.PI(acquisition_params),
            acq.UCB(acquisition_params),
            acq.EPS(acquisition_params),
            acq.TS(acquisition_params)
        ],
        "N": 100000,
        "xlabel": "Iteration",
        "ylabel": "Average Click Losses",
        "working_dir": [
            os.path.join(os.getcwd(), "output", "testclicks1d"),
            os.path.join(os.getcwd(), "output", "testclicks2d"),
            os.path.join(os.getcwd(), "output", "testclicks3d")
        ][dim-1]
    }
