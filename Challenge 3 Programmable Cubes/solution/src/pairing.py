from numba import njit
from solution.src.misc import *
from solution.src.pathfinding import *
import numpy as np
import random

def pair_colours(wi:np.ndarray,wt:np.ndarray,ci:np.ndarray,ti:np.ndarray,ct:np.ndarray,tt:np.ndarray):
    """ 
    pairs based on colours, does not consider distance
    """
    pairs = np.zeros(shape=(len(wi),4),dtype=np.int64)
    if not wi.size > 0:
        return pairs
    shuffled_wt = wt.copy()
    random.shuffle(shuffled_wt)
    pairs[:,0] = wi
    pairs[:,1] = shuffled_wt
    pairs[:,2] = ti[wi]
    pairs[:,3] = tt[shuffled_wt]
    for i in np.arange(len(wi)):
        if pairs[i,2] != pairs[i,3]:
            for j in np.arange(i+1,len(wi)):
                if pairs[i,2] == pairs[j,3]:
                    tmp_id = pairs[i,1]
                    tmp_type = pairs[i,3]
                    pairs[i,1] = pairs[j,1]
                    pairs[i,3] = pairs[j,3]
                    pairs[j,1] = tmp_id
                    pairs[j,3] = tmp_type
                    break
    return pairs

def pair_shuffled(wi:np.ndarray,wt:np.ndarray,ci:np.ndarray,ti:np.ndarray,ct:np.ndarray,tt:np.ndarray):
    """ 
    pairs based on colours, does not consider distance
    """
    pairs = np.zeros(shape=(len(wi),4),dtype=np.int64)
    if not wi.size > 0:
        return pairs
    shuffled_wt = wt.copy()
    random.shuffle(shuffled_wt)
    pairs[:,0] = wi
    pairs[:,1] = shuffled_wt
    pairs[:,2] = ti[wi]
    pairs[:,3] = tt[shuffled_wt]
    return pairs