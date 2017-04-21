__author__ = 'cipriancorneanu'

from sequential import Sequential
import numpy as np

class SequentialKalman(Sequential):
    def __init__(self, aligner, encoder=None,  mem=50):
        Sequential.__init__(self, aligner, encoder)
        self.Q = 0.01
        self.R = 0.01

        self.mem = mem

        self.xhat = None      # aposteri estimate of x
        self.P = None         # aposteri error estimate
        self.xhatminus = None # apriori estimate of x
        self.Pminus = None    # apriori error estimate
        self.K = None         # gain
        self.A = None         # update

        self.old_geometries = None

    def _predict_geometry(self, frames, indices):
        # Initialize old_geometries
        if self.old_geometries is None:
            self.old_geometries = np.zeros(self.geometries.shape + (1,))

        # Update old geometries
        if self.old_geometries.shape[-1]<self.mem:
            self.old_geometries = np.dstack((self.old_geometries, self.geometries))
        else:
            self.old_geometries = np.dstack((self.old_geometries[...,1:], self.geometries))

        self.xhat = np.zeros(self.old_geometries.shape)
        self.P = np.zeros(self.old_geometries.shape)
        self.xhatminus = np.zeros(self.old_geometries.shape)
        self.Pminus = np.zeros(self.old_geometries.shape)
        self.K = np.zeros(self.old_geometries.shape)
        self.U = np.zeros(self.geometries.shape)

        # Intial guesses
        self.xhat[...,0] = np.zeros(self.geometries.shape)
        self.P[...,0] = 0

        # Loop memory buffer and compute state estimation
        for k in range(1, self.old_geometries.shape[-1]):
            # Predict
            self.xhatminus[..., k] = self.U + self.xhat[..., k-1]
            self.Pminus[..., k] = self.P[..., k-1]+self.Q

            # Update
            self.K[...,k] = self.Pminus[...,k]/(self.Pminus[...,k]+self.R)
            self.xhat[...,k] = self.xhatminus[...,k]+self.K[...,k]*(self.old_geometries[...,k]-self.xhatminus[...,k])
            self.P[...,k] = (1-self.K[...,k])*self.Pminus[...,k]

            # Compute process update
            self.U = self.xhat[...,k] - self.xhat[...,k-1]

        return (self.U + self.xhat[...,-1])[indices]