from cascade import Cascade
from ..encoder.encoder import Encoder
from ..toolkit.procrustes import procrustes_generalized
import numpy as np


class CascadeBasic(Cascade):
    def __init__(self, regressor='linear', descriptor='sift_rotate'):
        Cascade.__init__(self, regressor, descriptor)
        self.mean_shape = None
        self.encoder = None

    # Parent class overrides
    # --------------------------------------------------

    def _initialize_method(self, images, ground_truth):
        self.mean_shape, _ = procrustes_generalized(ground_truth)
        self.mean_shape += np.mean(np.mean(ground_truth, axis=0), axis=0)[None, :]
        self.encoder = Encoder(self.num_landmarks, self.num_dimensions)

    def _initialize_instances(self, images):
        return np.tile(self.mean_shape.reshape((1, -1)), (images.shape[0], 1))

    def _rotate_targets(self, encoded, angles):
        decoded = self.encoder.decode_parameters(encoded)
        decoded[:, :, :2] = self._apply_rotations(decoded[:, :, :2], angles, center=False)
        return self.encoder.encode_parameters(decoded)

    def _train_step(self, images, ground_truth, params, mapping, i, args=None):
        raise NotImplementedError('_train_step not implemented for base class CascadeBasic')

    def _align_step(self, images, params, mapping, i, features=None, args=None):
        raise NotImplementedError('_align_step not implemented for base class CascadeBasic')
