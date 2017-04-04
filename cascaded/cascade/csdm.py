from cascade_basic import CascadeBasic
from cascade_parametric import CascadeParametric
from ..toolkit.pca import variation_modes
import numpy as np


class CascadeCsdmAbstract:
    def _train_step(self, images, ground_truth, params, mapping, i, args=None):
        # Read training parameters
        nb_shape = self.nb_shape if args is None else args.get('nb_shape', self.nb_shape)
        nb_feats = self.nb_feats if args is None else args.get('nb_feats', self.nb_feats)
        var = self.variance[i] if isinstance(self.variance, list) else self.variance

        # Select step-wise parameters if not uniform
        nb_shape = nb_shape[i] if isinstance(nb_shape, list) else nb_shape
        nb_feats = nb_feats[i] if isinstance(nb_feats, list) else nb_feats

        # Prepare shapes and rotations
        shapes = self.encoder.decode_parameters(params)
        rotations = self._get_angles(shapes)

        # Extract features
        descriptor = self.descriptor()
        descriptor.initialize(images, shapes[:, :, :2], mapping, args={'rotations': rotations})
        features, visibility = descriptor.extract(images, shapes[:, :, :2], mapping, args={'rotations': rotations})

        # Find features compression
        mean_features, pca_transform, _ = variation_modes(features, min_variance=var)
        features = np.dot(features - mean_features[None, :], pca_transform)

        # Prepare rotated target deltas
        targets = self._rotate_targets(ground_truth[mapping, ...] - params, rotations)

        # Find two principal components of target deltas
        dtargs_mean, dtargs_tfm, _ = variation_modes(targets, n_bases=nb_shape)
        bases = np.concatenate((
            np.dot(targets - dtargs_mean, dtargs_tfm),
            features[:, :nb_feats]
        ), axis=1)

        # Learn regressor
        regressor = self.regressor()
        tr_preds = regressor.learn(bases, features, targets, mapping)
        return {'regressor': regressor, 'descriptor': {
            'descriptor': descriptor,
            'feats_mean': mean_features,
            'feats_tfm': pca_transform,
            'dtargs_mean': dtargs_mean,
            'dtargs_tfm': dtargs_tfm,
            'nb_feats': nb_feats,
        }}, self._rotate_targets(tr_preds, -rotations)

    def _align_step(self, images, params, mapping, i, features=None, args=None):
        descriptor = self.steps[i]['descriptor']
        args = {} if args is None else args

        # Prepare shapes and rotations
        shapes = self.encoder.decode_parameters(params)
        rotations = self._get_angles(shapes)

        # Extract features
        features = features if features is not None else np.dot(descriptor['descriptor'].extract(
            images,
            shapes[:, :, :2],
            mapping,
            args={'rotations': rotations}
        )[0].reshape((len(mapping), -1)) - descriptor['feats_mean'][None, :], descriptor['feats_tfm'])

        # Rotate target deltas if required
        dtargs = None if descriptor['dtargs_tfm'].shape[1] <= 0 else self._rotate_targets(
            args['target'][mapping, ...] - params, rotations
        )

        # Get bases
        bases = np.dot(
            dtargs - descriptor['dtargs_mean'], descriptor['dtargs_tfm']
        ) if descriptor['dtargs_tfm'].shape[1] > 0 else np.zeros((features.shape[0], 0), dtype=np.float32)
        bases = np.concatenate((bases, features[:, :descriptor['nb_feats']]), axis=1)

        # Apply regressor
        return params + self._rotate_targets(self.steps[i]['regressor'].apply(bases, features), -rotations), bases


class CascadeCsdm(CascadeCsdmAbstract, CascadeBasic):
    def __init__(self, regressor='metalinear', descriptor='sift_rotate', nb_shape=2, nb_feats=3, variance=0.9):
        CascadeBasic.__init__(self, regressor, descriptor)
        self.nb_shape, self.nb_feats, self.variance = nb_shape, nb_feats, variance


class CascadeCsdmParametric(CascadeCsdmAbstract, CascadeParametric):
    def __init__(self, regressor='metalinear', descriptor='sift_rotate', nb_shape=2, nb_feats=3, variance=0.9):
        CascadeParametric.__init__(self, regressor, descriptor)
        self.nb_shape, self.nb_feats, self.variance = nb_shape, nb_feats, variance
