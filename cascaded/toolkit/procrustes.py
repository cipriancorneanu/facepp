import numpy as np
import matplotlib.pyplot as plt
from linalg import transform_shapes


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, tform = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':
        # If reflection applied but not asked for, force another reflection
        if reflection != (np.linalg.det(T) < 0):
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b*np.dot(muY, T)

    return d, Z, {'rotation': T, 'scale': b, 'translation': c}


def procrustes_generalized(shapes, num_iter=5, verbose=True):
        # Get shapes dimension, append extra dimension (ones) for translation
        n_i, n_l, n_d = shapes.shape

        # Optimize transforms and mean shape
        m, tfms = None, [{
            'rotation': np.eye(n_d, dtype=np.float32),
            'scale': 1,
            'translation': np.mean(s, axis=0)
        } for s in shapes]

        # Display MSE of alignment
        if verbose:
            mse = np.mean((shapes - np.mean(shapes, axis=0)[None, :]) ** 2)
            print 'Generalized Procrustes Analysis error (@0): ' + str(mse)

        # Optimize transforms and mean shape
        a_shapes = shapes
        for it in range(num_iter):
            # Calculate mean
            m = np.mean(a_shapes, axis=0)
            m = (m - np.mean(m, axis=0)[None, :]) / np.mean([t['scale'] for t in tfms])

            # Re-align instances to mean
            for ir in range(n_i):
                _, _, tfms[ir] = procrustes(m, shapes[ir, ...], reflection=False)

            # Display MSE of alignment
            a_shapes = transform_shapes(shapes, tfms)
            if verbose:
                mse = np.mean((a_shapes - m[None, ...]) ** 2)
                print 'Generalized Procrustes Analysis error (@' + str(1+it) + '): ' + str(mse)

        # Return mean and transforms
        return m, tfms
