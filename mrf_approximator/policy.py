
from enum import Enum
import numpy as np
from .ipc import posterior2ratio

class MetaPolicy(Enum):
    ACQ = 0
    TS = 1
    EPS = 2
    ELSE = 3
   
    @staticmethod
    def getDec(meta, dec_map):
        return dec_map[meta]
    
    @staticmethod
    def getTrafficACQ(policy):
        def wrapper(N, y, var):
            i = np.argmax(policy.acq.acquire(y, var))
            retVal = np.zeros(len(y))
            retVal[i] = N
            return np.array(retVal).astype(np.int32)
        return wrapper

    @staticmethod
    def getTrafficEPS(policy):
        def wrapper(N, y, var):
            eps = policy.acq.params["eps"]
            greedy = N * (1 - eps)
            retVal = np.ones(len(y)) * ((N - greedy) / (N - 1))
            retVal[np.argmax(y)] = greedy
            return np.array(retVal).astype(np.int32)
        return wrapper

    @staticmethod
    def getTrafficTS(policy):
        def wrapper(N, y, var):
            weights = posterior2ratio(y, var)
            return np.array(N * weights).astype(np.int32)
        return wrapper

    @staticmethod
    def getTrafficUniform(policy):
        def wrapper(N, y, var):
            return np.array([N // len(y)] * len(y)).astype(np.int32)
        return wrapper

class Policy:
    policy_map = {"ei": MetaPolicy.ACQ,
                  "pi": MetaPolicy.ACQ,
                  "ucb": MetaPolicy.ACQ,
                  "ts": MetaPolicy.TS,
                  "eps": MetaPolicy.EPS}

    impression_map = {MetaPolicy.ACQ: MetaPolicy.getTrafficACQ, 
                     MetaPolicy.TS: MetaPolicy.getTrafficTS,
                     MetaPolicy.EPS: MetaPolicy.getTrafficEPS,
                     MetaPolicy.ELSE: MetaPolicy.getTrafficUniform}
    
    def __init__(self, acq):
        self.meta_policy = Policy.policy_map[str(acq)] if acq is not None else MetaPolicy.ELSE
        self.acq = acq

    def trafficGetter(self):
        # return a func(N, y, var)
        return Policy.impression_map[self.meta_policy](self)



