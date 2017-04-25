from regressor import Regressor
import numpy as np
from scipy import linalg


class RegressorMetalinear2(Regressor):
    def __init__(self):
        Regressor.__init__(self)
        self.n_features, self.n_targets, self.n_bases = (None, None, None)
        self.m_weights = None

    def learn(self, f_projs, inputs, targets, indices):
        self.n_features, self.n_targets, self.n_bases = inputs.shape[1], targets.shape[1], f_projs.shape[1]
        khatri_rprod = self._khatri_rao(inputs, f_projs)

        # Prepare helper matrices & calculate final regressor
        a = np.linalg.pinv(np.dot(np.transpose(khatri_rprod), khatri_rprod))
        b = np.dot(np.transpose(khatri_rprod), targets)
        self.m_weights = np.dot(a, b)

        # Calculate fit results_disfa with k-fold cross-validation
        preds, s_f, s_t = np.empty(targets.shape, dtype=np.float32), khatri_rprod.shape[1], self.n_targets
        for i in range(np.max(indices) + 1):
            i_valid = np.where(indices == i)
            f, t = khatri_rprod[i_valid, :].reshape((-1, s_f)), targets[i_valid, :].reshape((-1, s_t))

            # Calculate correction for A
            p = np.dot(a, np.transpose(f))
            cap = np.linalg.pinv(np.dot(f, p) + np.eye(len(i_valid)))
            a_corr = a + np.dot(p, np.dot(cap, np.transpose(p)))

            preds[i_valid, :] = np.dot(np.dot(f, a_corr), b - np.dot(np.transpose(f), t))

        return preds

    def apply(self, f_projs, inputs):
        khatri_rprod = self._khatri_rao(inputs, f_projs[:, :self.n_bases])
        return np.dot(khatri_rprod, self.m_weights)

    @staticmethod
    def _khatri_rao(features, bases):
        n_inst, n_features, n_bases = features.shape[0], features.shape[1], bases.shape[1]
        ret = np.empty((n_inst, (n_features+1)*(n_bases+1)), dtype=np.float32)

        ret[:, n_bases*(n_features+1):-1] = features
        ret[:, -1] = 1
        for i in range(n_bases):
            ret[:, i*(n_features+1):(i+1)*(n_features+1)-1] = features * bases[:, i][:, None]
            ret[:, (i+1)*(n_features+1)-1] = bases[:, i]

        return ret
