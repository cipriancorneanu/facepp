from cascade_basic import CascadeBasic
from cascade_parametric import CascadeParametric
from ..toolkit.pca import variation_modes
import numpy as np


class CascadeGsdmAbstract:
    def _train_step(self, images, ground_truth, params, mapping, i, args=None):
        nb_shape = self.nb_shape if args is None else args.get('nb_shape', self.nb_shape)
        nb_feats = self.nb_feats if args is None else args.get('nb_feats', self.nb_feats)

        # Get shapes and rotations
        shapes = self.encoder.decode_parameters(params)
        rotations = self._get_angles(shapes)

        # Extract features
        descriptor = self.descriptor()
        descriptor.initialize(images, shapes[:, :, :2], mapping, args={'rotations': rotations})
        features, visibility = descriptor.extract(images, shapes[:, :, :2], mapping, args={'rotations': rotations})

        # Find features compression
        feats_mean, feats_tfm, _ = variation_modes(features, min_variance=0.90)
        features = np.dot(features - feats_mean[None, :], feats_tfm)

        # Prepare rotated target deltas
        targets = self._rotate_targets(ground_truth[mapping, ...] - params, rotations)

        # Find two principal components of target deltas
        dtargs_mean, dtargs_tfm, _ = variation_modes(targets, n_bases=nb_shape)
        bases = np.concatenate((
            np.dot(targets - dtargs_mean, dtargs_tfm),
            features[:, :nb_feats]
        ), axis=1)

        # Train individual regressors
        reg_instances = np.sum((bases > 0) * (2 ** np.arange(nb_feats+nb_shape))[None, ::-1], axis=1)
        regressors = [None] * (2 ** bases.shape[1])
        tr_preds = np.zeros((len(mapping), ground_truth.shape[1]), dtype=np.float32)
        for i in range(2 ** bases.shape[1]):
            regressors[i] = self.regressor()
            tr_preds[reg_instances == i, :] = regressors[i].learn(
                features[reg_instances == i, :],
                targets[reg_instances == i, :]
            )

        # Save regressor structure
        return {'regressors': regressors, 'descriptor': {
            'descriptor': descriptor,
            'feats_mean': feats_mean,
            'feats_tfm': feats_tfm,
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

        # Apply specific regressor to each instance
        reg_instances = np.sum((bases > 0) * (2 ** np.arange(bases.shape[1]))[None, ::-1], axis=1)
        for j in range(2 ** bases.shape[1]):
            params[reg_instances == j, :] += self._rotate_targets(
                self.steps[i]['regressors'][j].apply(features[reg_instances == j, :]),
                -rotations[reg_instances == j]
            )

        return params, bases


class CascadeGsdm(CascadeGsdmAbstract, CascadeBasic):
    def __init__(self, regressor='linear', descriptor='sift_rotate', nb_shape=2, nb_feats=1):
        CascadeBasic.__init__(self, regressor, descriptor)
        self.nb_shape, self.nb_feats = nb_shape, nb_feats


class CascadeGsdmParametric(CascadeGsdmAbstract, CascadeParametric):
    def __init__(self, regressor='linear', descriptor='sift_rotate', nb_shape=2, nb_feats=1):
        CascadeParametric.__init__(self, regressor, descriptor)
        self.nb_shape, self.nb_feats = nb_shape, nb_feats
