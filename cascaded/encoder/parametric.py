from encoder import Encoder
import numpy as np

from ..toolkit.linalg import transform_shapes, get_rotation_angles, build_rotation_matrix
from ..toolkit.procrustes import procrustes


class EncoderParametric(Encoder):
    def __init__(self, num_landmarks, num_dimensions, shape_mean, shape_transform):
        Encoder.__init__(self, num_landmarks, num_dimensions)
        self.shape_mean = shape_mean
        self.shape_transform = shape_transform
        self.num_bases = self.shape_transform.shape[1]
        self.noise_covariance = None

    def set_num_bases(self, num_bases):
        self.num_bases = self.shape_transform.shape[1] if num_bases > self.shape_transform.shape[1] else num_bases

    def get_parameters_length(self):
        return self.num_bases + (4 if self.num_dimensions == 2 else 6)

    def encode_parameters(self, decoded):
        return self._encode_2d(decoded) if self.num_dimensions == 2 else self._encode_3d(decoded)

    def decode_parameters(self, encoded):
        n_spars = self.num_bases
        n_cpars = 4 if self.num_dimensions == 2 else 6

        # Prepare matrix for the decoded shapes
        decoded = self.shape_mean + np.dot(
            encoded[:, :n_spars],
            np.transpose(self.shape_transform[:, :self.num_bases])
        ).reshape((-1,) + self.shape_mean.shape)

        # Prepare transforms
        decoded = transform_shapes(decoded, [{
            'scale': p[-1],
            'rotation': build_rotation_matrix(p[-2]) if n_cpars == 4 else build_rotation_matrix(p[-4], p[-3], p[-2]),
            'translation': np.array([p[n_spars], p[n_spars + 1]] + ([] if n_cpars == 4 else [0]), dtype=np.float32),
         } for p in encoded], inverse=False)

        return decoded

    def transcode_parameters(self, decoded):
        return self.decode_parameters(self.encode_parameters(decoded))

    def model_noise(self, decoded):
        # Capture noise
        tfms = [procrustes(s, self.shape_mean, reflection=False)[2] for s in decoded]
        vars = transform_shapes(decoded, tfms, inverse=True) - self.shape_mean[None, ...]
        vars = np.reshape(vars, (vars.shape[0], -1))
        noise = vars - np.dot(vars, np.dot(self.shape_transform, np.transpose(self.shape_transform)))

        # Learn multivariate normal model for noise
        self.noise_covariance = np.cov(np.transpose(noise))

    def introduce_noise(self, decoded):
        tfms = [procrustes(s, self.shape_mean, reflection=False)[2] for s in decoded]
        vars = transform_shapes(decoded, tfms, inverse=True) - self.shape_mean[None, ...]
        vars = np.dot(
            np.reshape(vars, (vars.shape[0], -1)),
            np.dot(self.shape_transform, np.transpose(self.shape_transform))
        ) + np.random.multivariate_normal(
            np.zeros(self.num_landmarks*self.num_dimensions),
            self.noise_covariance,
            len(decoded)
        )

        # Return sequences with their original shape
        return transform_shapes(
            np.reshape(vars, (-1, self.num_landmarks, self.num_dimensions)) + self.shape_mean[None, ...],
            tfms,
            inverse=False
        )

    def encode_deltas(self, shapes_prev, shapes_curr):
        # Obtain delta encodings
        encoded_old = self.encode_parameters(shapes_prev)
        encoded_new = self.encode_parameters(shapes_curr)
        deltas = encoded_new - encoded_old

        # Make delta scales scale-invariant
        deltas[:, -1] /= encoded_old[:, -1]

        # Make delta translations scale-invariant
        l, r = (-4, -2) if self.num_dimensions == 2 else (-6, -4)
        deltas[:, l:r] /= encoded_old[:, -1, None]

        return deltas

    def decode_deltas(self, params_old, params_delta):
        # Undo delta scales scale invariance
        params_delta[:, -1] *= params_old[:, -1]

        # Undo delta translations scale invariance
        l, r = (-4, -2) if self.num_dimensions == 2 else (-6, -4)
        params_delta[:, l:r] *= params_old[:, -1, None]

        # Update parameters and decode
        return self.decode_parameters(params_old + params_delta)

    def _encode_2d(self, decoded):
        # Prepare matrix for the encodings and alignemt transforms
        encoded = np.empty((decoded.shape[0], self.num_bases + 4), dtype=np.float32)
        tfms = [procrustes(s, self.shape_mean, reflection=False)[2] for s in decoded]

        # Encode camera parameters
        for i, tfm in enumerate(tfms):
            encoded[i, -1] = tfm['scale']  # Encode scaling
            encoded[i, -2] = get_rotation_angles(tfm['rotation'])  # Encode rotation
            encoded[i, -4:-2] = tfm['translation']  # Encode translation

        # Encode shape parameters
        encoded[:, :self.num_bases] = np.dot((
            transform_shapes(decoded, tfms, inverse=True) - self.shape_mean[None, ...]
        ).reshape((decoded.shape[0], -1)), self.shape_transform[:, :self.num_bases])

        return encoded

    def _encode_3d(self, decoded):
        # Prepare matrix for the encodings and alignemt transforms
        encoded = np.empty((decoded.shape[0], self.num_bases + 6), dtype=np.float32)
        tfms = [procrustes(s, self.shape_mean, reflection=False)[2] for s in decoded]

        # Encode camera parameters
        for i, tfm in enumerate(tfms):
            encoded[i, -1] = tfm['scale']  # Encode scaling
            encoded[i, -4:-1] = get_rotation_angles(tfm['rotation'])  # Encode rotation
            encoded[i, -6:-4] = tfm['translation'][:2]  # Encode translation

        # Encode shape parameters
        encoded[:, :self.num_bases] = np.dot((
            transform_shapes(decoded, tfms, inverse=True) - self.shape_mean[None, :, :]
        ).reshape((decoded.shape[0], -1)), self.shape_transform[:, :self.num_bases])

        return encoded
