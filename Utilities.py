import numpy as np

def Frequencies(TM, nstates):
    """
    Determine average frequencies of states over set of trajectories TM.
    """
    # Histogram columns one by one:
    count = np.zeros(nstates)
    for j in range(TM.shape[1]):
        count = count + np.bincount(TM[:, j], minlength=nstates)
    count = count/(TM.shape[1]*TM.shape[0])
    return count