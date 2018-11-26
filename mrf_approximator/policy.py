
from enum import Enum
import numpy as np
from .ipc import posterior2ratio


class MetaPolicy(Enum):
    ACQ = 0
    TS = 1
    EPS = 2
    ELSE = 3

    @staticmethod
    def traffic_handler_acq(policy, **kwargs):
        def wrapper(N, y, var):
            i = np.argmax(policy.acq.acquire(y, var, best=kwargs["best"], rsize=kwargs["rsize"]))
            ret = np.ones(len(y))
            ret[i] = N - len(y) + 1
            return np.array(ret).astype(np.int32)
        return wrapper

    @staticmethod
    def traffic_handler_eps(policy, **kwargs):
        def wrapper(N, y, var):
            eps = policy.acq.params["eps"]
            greedy = N * (1 - eps)
            ret = np.ones(len(y)) * ((N - greedy) / (len(y) - 1))
            ret[np.argmax(y)] = greedy
            return np.array(ret).astype(np.int32)
        return wrapper

    @staticmethod
    def traffic_handler_ts(policy, **kwargs):
        def wrapper(N, y, var):
            weights = posterior2ratio(y, var)
            return (weights * N).astype(np.int32)
            # i = np.argmax(weights)
            # ret = np.ones(len(y))
            # ret[i] = N - len(y) + 1
            # return np.array(ret).astype(np.int32)
        return wrapper

    @staticmethod
    def traffic_handler_uniform(policy, **kwargs):
        def wrapper(N, y, var):
            return np.array([N // len(y)] * len(y)).astype(np.int32)
        return wrapper


class Policy:
    meta_policy_map = {"ei": MetaPolicy.ACQ,
                       "pi": MetaPolicy.ACQ,
                       "ucb": MetaPolicy.ACQ,
                       "ts": MetaPolicy.TS,
                       "eps": MetaPolicy.EPS}

    scheduler_map = {MetaPolicy.ACQ: MetaPolicy.traffic_handler_acq,
                     MetaPolicy.TS: MetaPolicy.traffic_handler_ts,
                     MetaPolicy.EPS: MetaPolicy.traffic_handler_eps,
                     MetaPolicy.ELSE: MetaPolicy.traffic_handler_uniform}
    
    def __init__(self, acq):
        self.meta_policy = Policy.meta_policy_map[str(acq)] if acq is not None else MetaPolicy.ELSE
        self.acq = acq

    def traffic_getter(self, **kwargs):
        # return a func(N, y, var)
        return Policy.scheduler_map[self.meta_policy](self, best=kwargs["best"], rsize=kwargs["rsize"])

