__author__ = 'cipriancorneanu'

from ..frontalizer.frontalize import frontalize
from ..frontalizer.facial_feature_detector import get_landmarks
from ..frontalizer.camera_calibration import estimate_camera
from scipy.misc import imresize
import cv2
import numpy as np
import time

def affine_align(face, geom=None, geom_predictor=None, kpts=(range(36,41), range(42, 47), range(48,67))):
    '''
    Perform alignment based on affine transformation computed using three points, left eye, right eye, mouth.
    The landmarks corresponding to the three parts of the face have to be predefined.
    '''
    h, w = face.shape[0], face.shape[1]

    if not geom and not geom_predictor:
        raise Exception('You must either provide geometry or a geometry predictor')

    if not geom:
        # Estimate facial geometry of input face
        geom = np.squeeze(get_landmarks(face, geom_predictor))

    if len(geom)>0:
        ref_t = np.float32(np.asarray([[110, 100], [210, 100], [162, 200]])) # Reference size is (320, 320)
        act_t = np.asarray([np.mean(geom[k,:], axis=0) for k in kpts])

        T = cv2.getAffineTransform(act_t, ref_t)

        face = cv2.warpAffine(face, T, (320, 320))
        geom = np.transpose(np.dot(T, np.transpose(np.hstack((geom, np.ones((68,1)))))))

        # Get ROI around detected facial geometry and add border
        border = 20
        minx, miny = [int(x-border) for x in np.min(geom, axis=0)]
        maxx, maxy = [int(x+border) for x in np.max(geom, axis=0)]

        # Resize to input size
        minx, maxx, miny, maxy = max(0,minx), min(face.shape[1], maxx), max(0,miny), min(face.shape[0], maxy)
        rh, rw = h/float(maxx-minx), w/float(maxy-miny)
        aface = imresize(face[miny:maxy, minx:maxx, ...], (h,w))
        ageom = (geom - (minx, miny)) * (rh, rw)

        return (aface, ageom)
    else:
        print 'Geometry could not be estimated.'
        return (face, np.zeros((68,2)))

def sym_align(face, model3D, eyemask, shape_predictor):
    '''
    Perform alignment based on Hassner, Tal, et al. "Effective face frontalization in unconstrained images."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
    Code forked from original repo: https://github.com/dougsouza/face-frontalization
    '''
    h, w = face.shape[0], face.shape[1]
    # Estimate facial geometry of input face
    geom = np.squeeze(get_landmarks(face, shape_predictor))

    # Perform camera calibration according to the first face detected
    try:
        proj_matrix, camera_matrix, rmat, tvec = estimate_camera(model3D, geom)

        # If one channel expand to 3 channels
        if len(face.shape) == 2:
            face = face.reshape(face.shape[0],face.shape[0],1).repeat(3,2)

        # Perform frontalization
        frontal_raw, frontal_sym, frontalized = frontalize(face, proj_matrix, model3D.ref_U, eyemask)

        if not frontalized:
            # Rely on affine alignment only
            face, geom = affine_align(face, geom=geom, geom_predictor=None)
        else:
            # Estimate facial geometry of frontalized face
            geom = np.squeeze(get_landmarks(frontal_sym, shape_predictor))
            face = frontal_sym

        # Get ROI around detected facial geometry and add border
        border = 10
        minx, miny = [int(x-border) for x in np.min(geom, axis=0)]
        maxx, maxy = [int(x+border) for x in np.max(geom, axis=0)]

        # Resize to input size
        rh, rw = h/float(maxx-minx), w/float(maxy-miny)
        aface = imresize(face[miny:maxy, minx:maxx, ...], (h,w))

        ageom = (geom - (minx, miny)) * (rh, rw)

    except Exception as e:
        print e
        return np.zeros((h, w, 3)), np.zeros((68,2))

    return aface, ageom

def batch_align(faces, model3D, eyemask, predictor):
    '''
    Align/frontalize batch of images
    '''

    n_faces = len(faces)
    if len(faces.shape)==4:
        afaces = np.zeros((n_faces, faces.shape[1], faces.shape[2], faces.shape[3]), dtype=np.uint8)
    elif len(faces.shape)==3:
        afaces = np.zeros((n_faces, faces.shape[1], faces.shape[2]), dtype=np.uint8)
    else:
        raise('Input format not correct')

    ageoms = np.zeros((n_faces,68,2))

    for i in range(0, n_faces):
        start_time = time.time()

        img = np.squeeze(faces[i,...])

        # Align face and geometry
        aface, ageom = sym_align(img, model3D, eyemask, predictor)

        if len(faces.shape)==4:
            afaces[i,...], ageoms[i,...] = np.squeeze(aface), ageom
        elif len(faces.shape)==3:
            afaces[i,...], ageoms[i,...] = np.squeeze(aface[:,:,0]), ageom

        print '     Alignining face {}/{}: {}'.format(i, n_faces, (time.time() - start_time))

    return afaces, ageoms

def align(i, face, model3D, eyemask, predictor, do_frontalize=True, verbose=False):
    # Align face and geometry
    if do_frontalize:
        aface, ageom = sym_align(face, model3D, eyemask, predictor)
    else:
        aface, ageom = affine_align(face, geom=None, geom_predictor=predictor)

    if verbose and i%20==0:
        print '         Alignining face {}'.format(i)

    return aface, ageom
