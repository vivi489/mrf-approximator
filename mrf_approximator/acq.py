
from scipy.stats import norm
import numpy as np
import abc 


class ACQ(abc.ABC):
    """
    ### example params ###
    acquisition_params = {
        "par": 0.01,
        "dim": 1,
        "eps": 0.2,
        "delta": 0.9
    }
    """
    def __init__(self, params):
        self.params = params
        self.name = None

    @abc.abstractmethod
    def acquire(self, mu, var, best=None, rsize=0):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class EI(ACQ):
    def __init__(self, params):
        super(EI, self).__init__(params)
        
    def acquire(self, mu, var, best=None, rsize=0):
        par = self.params["par"]
        sigma = np.sqrt(var)
        if best is None:
            z = (mu - par) / sigma
        else:
            z = (mu - best - par) / sigma
        return z * sigma * norm.cdf(z) + sigma * norm.pdf(z)
    
    def __str__(self):
        return "ei"


class PI(ACQ):
    def __init__(self, params):
        super(PI, self).__init__(params)
        self.name = "pi"
        
    def acquire(self, mu, var, best=None, rsize=0):
        par = self.params["par"]
        sigma = np.sqrt(var)
        if best is None:
            z = (mu - par) / sigma
        else:
            z = (mu - best - par) / sigma
        return norm.cdf(z)
    
    def __str__(self):
        return "pi"


class UCB(ACQ):
    def __init__(self, params):
        super(UCB, self).__init__(params)
        self.name = "ucb"
        
    def _getbeta(self, rsize):
        # delta must be in (0, 1)
        rsize += 1
        beta = 2 * np.log(self.params["dim"] * ((rsize * np.pi) ** 2) / (6 * self.params["delta"]))
        return beta
    
    def acquire(self, mu, var, best=None, rsize=0):
        beta = self._getbeta(rsize)
        sigma = np.sqrt(var)
        return mu + sigma * np.sqrt(beta)
    
    def __str__(self):
        return "ucb"


class TS(ACQ):
    def __init__(self, params):
        super(TS, self).__init__(params)
        self.name = "ts"
        
    def acquire(self, mu, var, best=None, rsize=0):
        sigma = np.sqrt(var)
        return np.random.normal(mu, sigma)
    
    def __str__(self):
        return "ts"
        

class EPS(ACQ):
    def __init__(self, params):
        super(EPS, self).__init__(params) 
        self.name = "eps"

    def acquire(self, mu, var, best=None, rsize=0):
        if np.random.rand() <= self.params["eps"]:
            return np.random.rand(len(mu))
        else:
            ret = np.zeros(len(mu))
            ret[np.argmax(mu)] = 1
            return ret
    
    def __str__(self):
        return "eps"
