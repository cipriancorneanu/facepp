from sequential import Sequential
import numpy as np
import random


class SequentialStub(Sequential):
    def __init__(self, aligner, encoder=None):
        Sequential.__init__(self, aligner, encoder=encoder)

    # Parent class access points
    # --------------------------------------------------

    def _initialize_memory(self):
        pass

    def _predict_geometry(self, frames, indices):
        return np.copy(self.geometries[indices, ...])
