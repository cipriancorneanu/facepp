from ..encoder.encoder import Encoder
from ..encoder.parametric import EncoderParametric
from ..toolkit.crop import crop_resize
import numpy as np
from scipy.ndimage.interpolation import zoom
import cPickle
import os


class Sequential:
    def __init__(self, aligner, encoder=None):
        num_landmarks = aligner['num_landmarks'] if isinstance(aligner, dict) else aligner.num_landmarks
        num_dimensions = aligner['num_dimensions'] if isinstance(aligner, dict) else aligner.num_dimensions

        self.aligner = aligner
        self.encoder = Encoder(num_landmarks, num_dimensions) if encoder is None else encoder
        self.geometries = None

    def save(self, file):
        # Check path exists, create otherwise
        directory = '/'.join(file.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Pickle object
        cPickle.dump(self, open(file + '_object.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        model = cPickle.load(open(file + '_object.pkl', 'rb'))
        return model

    def train(self, sequences, geometries, save_file=None):
        # If there is a _train function, train sequential aligner
        if hasattr(self, '_train'):
            getattr(self, '_train')(sequences, geometries, save_file=save_file)

    def test(self, sequences, geometries):
        # Initialize return geometries, prepare results matrices
        self.geometries = np.copy(geometries)
        geometries = [np.concatenate((g[None, ...], np.empty(
            (s.shape[0]-1,)+g.shape, dtype=np.float32
        )), axis=0) for g, s in zip(geometries, sequences)]

        # Align the remaining frames
        seq_lens = np.array([s.shape[0] for s in sequences])
        for i in range(1, max(seq_lens)):
            unfinished = np.where(i < seq_lens)[0]
            frames = [sequences[s][i, ...] for s in unfinished]
            aligns = self._align(frames, unfinished)
            for j, a in zip(unfinished, aligns):
                geometries[j][i, ...] = a

        # Return geometries
        return [g[1:, ...] for g in geometries]

    def unittest_aligner(self, geometries):
        predictor = getattr(self, '_predict_geometry')

        # Prepare geometries of first frames
        self.geometries = np.array([g[0, ...] for g in geometries])

        # Prepare results matrices & sequence lengths
        predictions = [np.empty((g.shape[0]-1,)+g.shape[1:], dtype=np.float32) for g in geometries]
        seq_lens = np.array([g.shape[0] for g in geometries])

        # Align the remaining frames
        for i in range(1, max(seq_lens)):
            # Select unfinished sequences
            unfinished = np.where(i < seq_lens)[0]

            # Perform predictions
            aligns = predictor(None, unfinished)
            for j, a in zip(unfinished, aligns):
                predictions[j][i-1, ...] = a

            # Update ground truth shapes
            self.geometries[unfinished, ...] = np.array([geometries[g][i, ...] for g in unfinished])

        # Return geometries
        return predictions

    def _align(self, frames, indices):
        # Predict and Fine-tune geometries to frames
        self.geometries[indices, ...] = getattr(self, '_predict_geometry')(frames, indices)
        self.geometries[indices, ...] = self._adjust_geometry(frames, self.geometries[indices, ...])
        return self.geometries[indices, ...]

    def _adjust_geometry(self, frames, geometries):
        crops, landmarks, transforms = Sequential._crop_roi(frames, geometries)
        predictions = self.aligner.align(crops, augmenter=landmarks.reshape((landmarks.shape[0], -1)))[0]
        return np.array([np.dot(p, t)[:, :-1] for p, t in zip(np.pad(
            predictions.reshape(geometries.shape),
            pad_width=((0, 0), (0, 0), (0, 1)),
            mode='constant',
            constant_values=1
        ), transforms)])

    @staticmethod
    def _crop_roi(frames, geometries):
        n_inst = geometries.shape[0]
        im, lm, tf = (
            np.empty((n_inst, 200, 200), dtype=np.uint8),
            np.empty(geometries.shape, dtype=np.float32),
            np.empty((n_inst, 3, 3), dtype=np.float32)
        )

        for i, (f, g) in enumerate(zip(frames, geometries)):
            im[i, ...], lm[i, ...], tf[i, ...] = crop_resize(f, g)

        return im, lm, tf