__author__ = 'cipriancorneanu'

from frontalizer.frontalize import frontalize
from frontalizer.frontalize import  ThreeD_Model
import frontalizer.facial_feature_detector as feature_detection
import frontalizer.camera_calibration as calib
import frontalizer.check_resources as check

import scipy.io as io
from scipy.misc import imresize
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time


def affine_align(face, geom, kpts=(range(36,41), range(42, 47), range(48,67))):
    '''
    Perform alignment based on affine transformation computed using three points, left eye, right eye, mouth.
    The landmarks corresponding to the three parts of the face have to be predefined.
    '''

    ref_t = np.float32(np.asarray([[110, 100], [210, 100], [162, 200]])) # hard-coded. To change!
    act_t = np.asarray([np.mean(geom[k,:], axis=0) for k in kpts])

    T = cv2.getAffineTransform(act_t, ref_t)

    aface = cv2.warpAffine(face, T, (320, 320))
    ageom = np.transpose(np.dot(T, np.transpose(np.hstack((geom, np.ones((68,1)))))))

    return aface, ageom

def sym_align(face, model3D):
    '''
    Perform alignment based on Hassner, Tal, et al. "Effective face frontalization in unconstrained images."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
    Code forked from original repo: https://github.com/dougsouza/face-frontalization
    '''
    h, w = face.shape[0], face.shape[1]
    # Estimate facial geometry of input face
    geom = np.squeeze(feature_detection.get_landmarks(face))

    # Perform camera calibration according to the first face detected
    try:
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, geom)

        # Load mask to exclude eyes from symmetry
        eyemask = np.asarray(io.loadmat('frontalizer/frontalization_models/eyemask.mat')['eyemask'])

        # If one channel expand to 3 channels
        if len(face.shape) == 2:
            face = face.reshape(face.shape[0],face.shape[0],1).repeat(3,2)

        # Perform frontalization
        frontal_raw, frontal_sym, frontalized = frontalize(face, proj_matrix, model3D.ref_U, eyemask)

        if not frontalized:
            # Rely on affine alignment only
            face, geom = affine_align(face, geom)
        else:
            # Estimate facial geometry of frontalized face
            geom = np.squeeze(feature_detection.get_landmarks(frontal_sym))
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
        return np.zeros((224, 224, 3)), np.zeros((68,2))

    return aface, ageom

def batch_align(faces, debug=False):
    '''
    Align/frontalize batch of images
    '''
    check.check_dlib_landmark_weights()

    # Load detections performed by dlib library on 3D model and reference image
    this_path = os.path.dirname(os.path.abspath(__file__))
    model3D = ThreeD_Model(this_path + "/frontalizer/frontalization_models/model3Ddlib.mat", 'model_dlib')

    n_faces = len(faces)
    afaces = np.zeros((n_faces, faces.shape[1], faces.shape[2]), dtype=np.uint8)
    ageoms = np.zeros((n_faces,68,2))

    for i in range(0, n_faces, 40):
        start_time = time.time()

        img = np.squeeze(faces[i,...])

        # Align face and geometry
        aface, ageom = sym_align(img, model3D)

        afaces[i,...], ageoms[i,...] = np.squeeze(aface[:,:,0]), ageom

        print '     Alignining face {}/{}: {}'.format(i, n_faces, (time.time() - start_time))

        if debug:
            plt.imshow(afaces[i, ...])
            plt.scatter(ageoms[i, :, 0], ageoms[i, :, 1])
            plt.savefig('/Users/cipriancorneanu/Research/data/fake/aligned_faces/final_'+str(i))
            plt.clf()

    return afaces, ageoms