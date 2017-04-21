import numpy as np
from sklearn.decomposition import PCA


def impute_variables(bases, instances, masks):
    r_data = (instances * masks).astype(np.float32, copy=False)
    tfm = np.dot(bases, np.transpose(bases))
    n_masks = ~masks

    for i in range(10):
        r_data[n_masks] = np.dot(r_data, tfm)[n_masks]

    return r_data


def variation_modes(data, mean=None, min_variance=0.95, n_bases=None, mask=None, sampling_size=None):
    # Sub-sample data if required
    apply_sampling = sampling_size is not None and sampling_size < data.shape[0]
    data = data[np.random.choice(data.shape[0], size=sampling_size), :] if apply_sampling else data

    # If not considering missing values & dataset small enough, use fast PCA implementation from sklearn
    if mask is None and np.prod(data.shape) < 500 * (10 ** 12):
        mean = np.mean(data, axis=0) if mean is None else mean
        pca = PCA(n_components=min_variance if n_bases is None else n_bases).fit(data - mean[None, :])
        return mean, np.transpose(pca.components_), pca.explained_variance_

    # Set missing data to 0, prepare floats mask
    f_mask = None
    if mask is not None:
        f_mask = mask.astype(np.float32)
        data[~mask] = 0

    # Calculate and subtract mean from samples
    v_size = data.shape[0] if mask is None else np.sum(f_mask, axis=0)
    mean = np.sum(data, axis=0) / v_size if mean is None else mean
    data = data - mean[None, :]
    if mask is not None:
        data[~mask] = 0

    # Calculate covariances matrix and decompose
    m_size = (data.shape[0] if mask is None else np.dot(np.transpose(f_mask), f_mask)) - 1
    covariances = np.dot(np.transpose(data), data) / m_size
    u, s, _ = np.linalg.svd(covariances)

    n_bases = np.where(np.cumsum(s / np.sum(s)) >= min_variance)[0][0]+1 if n_bases is None else n_bases
    return mean, u[:, :n_bases], s[:n_bases]
