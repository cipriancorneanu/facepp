from cascade import Cascade
from ..encoder.parametric import EncoderParametric
from ..toolkit.linalg import transform_shapes
from ..toolkit.procrustes import procrustes_generalized
from ..toolkit.pca import variation_modes
import numpy as np


class CascadeParametric(Cascade):
    def __init__(self, regressor='linear', descriptor='sift_rotate'):
        Cascade.__init__(self, regressor, descriptor)
        self.mean_shape, self.mean_params, self.shp_transform = None, None, None
        self.encoder = None

    # Parent class overrides
    # --------------------------------------------------

    def _initialize_method(self, images, ground_truth):
        # Get differences of aligned shapes from mean shape
        self.mean_shape, tfms = procrustes_generalized(ground_truth)
        diffs = (transform_shapes(
            ground_truth, tfms
        ) - self.mean_shape[None, :, :]).reshape((ground_truth.shape[0], -1))

        # Prepare shape compression transform and encoder/decoder
        _, self.shp_transform, _ = variation_modes(diffs, min_variance=0.99)
        self.encoder = EncoderParametric(
            self.num_landmarks, self.num_dimensions, self.mean_shape, self.shp_transform
        )

        # Encode initialization shape
        self.mean_params = self.encoder.encode_parameters(
            (self.mean_shape + np.mean(ground_truth, axis=(0, 1))[None, :])[None, ...]
        ).flatten()

    def _initialize_instances(self, images):
        return np.tile(self.mean_params[None, :], (images.shape[0], 1))

    def _rotate_targets(self, encoded, angles):
        encoded, n_spars = np.copy(encoded), self.shp_transform.shape[1]
        encoded[:, n_spars:n_spars+2] = self._apply_rotations(
            encoded[:, None, n_spars:n_spars+2], angles, center=False
        )[:, 0, :]

        return encoded

    def _train_step(self, images, ground_truth, params, mapping, i, args=None):
        raise NotImplementedError('_train_step not implemented for base class CascadeParametric')

    def _align_step(self, images, params, mapping, i, features=None, args=None):
        raise NotImplementedError('_align_step not implemented for base class CascadeParametric')
