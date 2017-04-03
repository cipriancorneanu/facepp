__author__ = 'cipriancorneanu'

from cascaded.toolkit import procrustes, pca, linalg
import numpy as np
import cPickle
from processor.partitioner import concat
import os

def encode_parametric(uncoded):
    # Procrustes
    _, tfms = procrustes.procrustes_generalized(uncoded, num_iter=5, verbose=True)

    # Transform
    aligned_uncoded = np.reshape(linalg.transform_shapes(uncoded, tfms, inverse=True),
                                (uncoded.shape[0], -1))

    # PCA
    mean, T, _ = pca.variation_modes(aligned_uncoded, min_variance=0.99)

    # Prepare coded
    coded = np.empty((uncoded.shape[0], T.shape[1] + 4), dtype=np.float32)

    # Encode rigid parameters
    for i, tfm in enumerate(tfms):
        coded[i, -1] = tfm['scale']  # Encode scaling
        coded[i, -2] = linalg.get_rotation_angles(tfm['rotation'])  # Encode rotation
        coded[i, -4:-2] = tfm['translation']  # Encode translation

    # Encode non-rigid parameters
    coded[:,:T.shape[1]] = np.dot(aligned_uncoded - mean[None, :], T)

    return coded, mean, T

def decode_parametric(coded, mean, T):
    num_bases = coded.shape[1]-4

    print 'Num bases: ' + str(num_bases)

    # Decode non-rigid
    decoded = (mean + np.dot(
        coded[:, :num_bases],
        np.transpose(T[:, :num_bases])
    )).reshape((coded.shape[0], -1, 2))

    # Decode rigid
    decoded = linalg.transform_shapes(decoded, [{
        'scale': p[-1],
        'rotation': linalg.build_rotation_matrix(p[-2]),
        'translation': np.array([p[-4:-2]], dtype=np.float32),
     } for p in coded], inverse=False)

    return decoded

def transcode_parametric(uncoded):
    return decode_parametric(*encode_parametric(uncoded))

def encode_nonparametric(uncoded):
    return np.reshape(uncoded,(uncoded.shape[0], -1))

def decode_nonparametric(coded):
    pass

def transcode_nonparametric(uncoded):
    return decode_nonparametric(*encode_nonparametric(uncoded))

def load_parametric(path, fname):
    if os.path.isfile(path+fname):
        return cPickle.load(open(path+fname, 'rb'))[:,...]
    else:
        ufname = fname.split('.')[0].split('_')[0]
        if os.path.isfile(path+ufname):
            uncoded = cPickle.load(open(path+ufname+'.pkl', 'rb'))['landmarks']
            encoded, _, _ = encode_parametric(uncoded)
            cPickle.dump(encoded, open(path+fname, 'wb'), cPickle.HIGHEST_PROTOCOL)
            return encoded[:,:11]

if __name__ ==  '__main__':
    # Load data
    path = '/Users/cipriancorneanu/Research/data/disfa/'
    dt = cPickle.load(open(path+'disfa.pkl', 'rb'))

    geom, _ = concat(dt['landmarks'])
    enc_geom, _, _ = encode_parametric(geom)

    cPickle.dump(enc_geom, open(path+'disfa_lms_enc.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)