__author__ = 'Douglas'

import dlib
import os
import numpy as np

this_path = os.path.dirname(__file__)

def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy


def get_landmarks(img, predictor):
    # if not automatically downloaded, get it from:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    predictor_path = this_path + "/dlib_models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()

    lmarks = []
    dets = detector(img, 1)
    shapes = []
    for k, det in enumerate(dets):
        shape = predictor(img, det)
        shapes.append(shape)
        xy = _shape_to_np(shape)
        lmarks.append(xy)

    lmarks = np.asarray(lmarks, dtype='float32')

    return lmarks