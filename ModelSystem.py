import numpy as np
import scipy.linalg as scl
import scipy.stats as sts

import msmtools.analysis as ma
import msmtools.generation as mg


class MS:
    def __init__(self):
        # Define the number of states:
        self.n = 6
        # Define the rate matrix:
        self.Q = np.array([[-0.5, 0.5, 0, 0, 0, 0],
                           [0.08, -0.1, 0.02, 0, 0, 0],
                           [0, 0.02, -0.1, 0.08, 0, 0],
                           [0, 0, 0.09, -0.1, 0.01, 0],
                           [0, 0, 0, 0.01, -0.1, 0.09],
                           [0, 0, 0, 0, 0.4, -0.4]])
        # Get the transition matrix:
        self.T = scl.expm(self.Q)
        # Get the eigenvalues and eigenvectors:
        self.R, self.d, self.L = self.ComputeRDLDecomp()
        # Get the stationary distribution:
        self.stat = ma.stationary_distribution(self.T)
        # Get the timescales:
        self.ts = -1.0/np.log(self.d)

    def TransitionMatrix(self):
        """
        Returns:
        --------
        T: ndarray, the n-by-n transition matrix of the system.
        """
        return self.T

    def StatDist(self):
        """
        Returns:
        --------
        pi: ndarray, the stationary distribution of the system.
        """
        return self.stat

    def Simulate(self, N, p0):
        """
        Simulate trajectory of length N started from discrete distribution p0
        Parameters:
        -----------
        N. int,
            length of trajectory
        p0: ndarray,
            discrete distribution from which the starting state is drawn.
        """
        # Draw starting state:
        p0d = sts.rv_discrete(values=(np.arange(self.n, dtype=int), p0))
        s0 = p0d.rvs(size=1)
        # Generate trajectory:
        traj = mg.generate_traj(self.T, N, s0)
        return traj

    def Eigenvalues(self):
        """
        Returns k dominant eigenvalues and eigenvectors of the transition matrix.
        Parameters:
        -----------
        k: int, the number of dominant eigenvalues and eigenvectors to be computed.

        """
        return self.d

    def Eigenvectors(self):
        """
        Compute the eigenvectors of the transition matrix.
        """
        return self.R, self.L

    def Timescales(self):
        return self.ts

    def ComputeRDLDecomp(self):
        """
        Computes the RDL-decomposition of the transition matrix.
        """
        R, d, L = ma.rdl_decomposition(self.T, reversible=True)
        return (R, np.diag(d), L)