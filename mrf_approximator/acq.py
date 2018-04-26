
from scipy.stats import norm
import numpy as np
import abc 

class ACQ:
    __metaclass__ = abc.ABCMeta
    def __init__(self, params):
        self.params = params
        self.name = None

    @abc.abstractmethod
    def acquire(self, mu, var, best=None, rsize=0):
        pass
    

class EI(ACQ):
    def __init__(self, params):
        super(EI, self).__init__(params)
        self.name = "ei"
        
    def acquire(self, mu, var, best=None, rsize=0):
        par = self.params["par"]
        sigma = np.sqrt(var)
        if best is None:
            z = (mu - par) / sigma
        else:
            z = (mu - best - par) / sigma
        return z * sigma * norm.cdf(z) + sigma * norm.pdf(z)

class PI(ACQ):
    def __init__(self, params):
        super(PI, self).__init__(params)
        self.name = "pi"
        
    def acquire(self, mu, var, best=None, rsize=0):
        par = self.params["par"]
        sigma = np.sqrt(var)
        if best is None:
            best = 0
        if best is None:
            z = (mu - par) / sigma
        else:
            z = (mu - best - par) / sigma
        return norm.cdf(z)

class UCB(ACQ):
    def __init__(self, params):
        super(UCB, self).__init__(params)
        self.name = "ucb"
        
    def _getBeta(self, rsize):
        #delta must be in (0, 1)
        rsize += 1
        #print(self.params["dim"], rsize, np.pi, self.params["delta"])
        beta = 2 * np.log(self.params["dim"] * ((rsize * np.pi) ** 2) / (6 * self.params["delta"]))
        return beta
    
    def acquire(self, mu, var, best=None, rsize=0):
        beta = self._getBeta(rsize)
        sigma = np.sqrt(var)
        return mu + sigma * np.sqrt(beta)
 
class TS(ACQ):
    def __init__(self, params):
        super(TS, self).__init__(params)
        self.name = "ts"
        
    def acquire(self, mu, var, best=None, rsize=0):
        sigma = np.sqrt(var)
        return np.random.normal(mu, sigma)
        

class EPS(ACQ):
    def __init__(self, params):
        super(EPS, self).__init__(params) 
        self.name = "eps"

    def acquire(self, mu, var, best=None, rsize=0):
        if np.random.rand() <= self.params["eps"]:
            return np.random.rand(len(mu))
        else:
            retVal = np.zeros(len(mu))
            retVal[np.argmax(mu)] = 1
            return retVal
    
    