import numpy as np
import math


def transform_shapes(shapes, transforms, inverse=False):
    aligned = np.empty(tuple(shapes.shape), dtype=np.float32)

    if inverse:
        for i, (s, t) in enumerate(zip(shapes, transforms)):
            aligned[i, ...] = np.dot(s - t['translation'][None, :], np.transpose(t['rotation']) / t['scale'])
    else:
        for i, (s, t) in enumerate(zip(shapes, transforms)):
            aligned[i, ...] = np.dot(s, t['scale']*t['rotation']) + t['translation'][None, :]

    return aligned


def build_rotation_matrix(roll, pitch=None, yaw=None):
    sc, cc = math.sin(roll), math.cos(roll)
    if pitch is None:
        return np.array([[cc, -sc], [sc, cc]], dtype=np.float32)

    sa, ca = math.sin(yaw), math.cos(yaw)
    sb, cb = math.sin(pitch), math.cos(pitch)
    return np.array([
        [ca*cb, ca*sb*sc-sa*cc, ca*sb*cc+sa*sc],
        [sa*cb, sa*sb*sc+ca*cc, sa*sb*cc-ca*sc],
        [-sb,   cb*sc,          cb*cc],
    ], dtype=np.float32)


def get_rotation_angles(matrix):
    return np.array([
        math.atan2(matrix[2, 1], matrix[2, 2]),
        math.atan2(-matrix[2, 0], math.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2)),
        math.atan2(matrix[1, 0], matrix[0, 0]),
    ], dtype=np.float32) if matrix.shape[0] == 3 else np.arctan2(matrix[1, 0], matrix[1, 1])