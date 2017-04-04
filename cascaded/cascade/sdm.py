from cascade_basic import CascadeBasic
from cascade_parametric import CascadeParametric
from ..toolkit.pca import variation_modes
import numpy as np


class CascadeSdmAbstract:
    def _train_step(self, images, ground_truth, params, mapping, i, args=None):
        # Calculate shapes and rotations
        shapes = self.encoder.decode_parameters(params)
        rotations = self._get_angles(shapes)

        # Extract features
        descriptor = self.descriptor()
        descriptor.initialize(images, shapes[:, :, :2], mapping, args={'rotations': rotations})
        features, visibility = descriptor.extract(images, shapes[:, :, :2], mapping, args={'rotations': rotations})

        mean_features, pca_transform, _ = variation_modes(features, min_variance=0.97)
        features = np.dot(features - mean_features[None, :], pca_transform)

        # Prepare rotated targets
        targets = self._rotate_targets(ground_truth[mapping, ...] - params, rotations)

        # Learn regressor
        regressor = self.regressor()
        tr_preds = regressor.learn(features, targets)
        return {'regressor': regressor, 'descriptor': {
            'descriptor': descriptor,
            'mean_features': mean_features,
            'pca_transform': pca_transform,
        }}, self._rotate_targets(tr_preds, -rotations)

    def _align_step(self, images, params, mapping, i, features=None, args=None):
        descriptor = self.steps[i]['descriptor']

        # Calculate shapes and rotations
        shapes = self.encoder.decode_parameters(params)
        rotations = self._get_angles(shapes)

        # Extract features
        features = features if features is not None else np.dot(descriptor['descriptor'].extract(
            images,
            shapes[:, :, :2],
            mapping,
            args={'rotations': rotations}
        )[0].reshape((len(mapping), -1)) - descriptor['mean_features'][None, :], descriptor['pca_transform'])

        # Apply regressor
        return params + self._rotate_targets(self.steps[i]['regressor'].apply(features), -rotations)


class CascadeSdm(CascadeSdmAbstract, CascadeBasic):
    def __init__(self, regressor='linear', descriptor='sift_rotate'):
        CascadeBasic.__init__(self, regressor, descriptor)


class CascadeSdmParametric(CascadeSdmAbstract, CascadeParametric):
    def __init__(self, regressor='linear', descriptor='sift_rotate'):
        CascadeParametric.__init__(self, regressor, descriptor)
