__author__ = 'cipriancorneanu'

import numpy as np
import scipy.misc
import dlib


def extract(im, lm, extension=1.3, size=200):
    # Crop and resize by keeping aspect ratio
    im, lm, t1 = _crop(im, lm, extension)
    im, lm, t2 = _resize(im, lm, size)
    return im, lm, np.dot(t2, t1)


def _crop(im, shape, extension=1.3):
    # Compute bounding box around shape
    bbox = _extend_rect([int(min(shape[:, 0])), int(min(shape[:, 1])),
                         int(max(shape[:, 0])), int(max(shape[:, 1]))], extension)

    # Bbox inside image
    ibbox = [int(max(0, bbox[0])), int(max(0, bbox[1])),
             min(im.shape[0], bbox[2]), min(im.shape[1], bbox[3])]

    # If inside get image if out put to zero
    imc = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1],
                    im.shape[2]), dtype=np.uint8)

    imc[0:ibbox[2]-ibbox[0], 0:ibbox[3]-ibbox[1], ...] = im[ibbox[0]
        :ibbox[2], ibbox[1]:ibbox[3], ...]

    return (
        imc,
        shape-bbox[:2],
        np.array([[1, 0, 0], [0, 1, 0], [bbox[0], bbox[1], 1]])
    )


def _resize(im, shape, size=200):
    # Get dominant dimension
    ddim = im.shape.index(max(im.shape))

    # Compute ratio and resize (square sizes only)
    r = float(size)/im.shape[ddim]
    im = scipy.misc.imresize(im, r)
    shape = shape*r

    # Thumbnail on smaller dimension
    margins = (np.array([size, size]) - np.array(im.shape[:2])) / 2
    imt = np.zeros((size, size, im.shape[2]), dtype=np.uint8)
    imt[margins[0]:margins[0]+im.shape[0],
        margins[1]:margins[1]+im.shape[1], ...] = im

    return (
        imt,
        shape + margins,
        np.array([[1 / r, 0, 0], [0, 1 / r, 0],
                  [-margins[0] / r, -margins[1] / r, 1]])
    )


def _extend_rect(rect, ext):
    # Compute extension params
    h2, w2 = int(ext*(rect[2]-rect[0])) / 2, int(ext*(rect[3]-rect[1])) / 2
    center_x, center_y = (rect[2]+rect[0])/2, (rect[3]+rect[1])/2

    # Extend bbox
    return [center_x-h2, center_y-w2, center_x+h2, center_y+w2]


def _pad(im, size):
    # Extend image to shape by adding black margins
    if im.shape < size:
        padded = np.zeros(size, dtype=im.dtype)
        padded[0:im.shape[0], 0:im.shape[1]] = im
        return padded


def square_bbox(shape):
    if np.amax(shape) > np.amin(shape):
        #print 'Shape: {}'.format(shape)
        # Compute rectangular bbox around shape
        bbox = [int(min(shape[:, 0])), int(min(shape[:, 1])),
                int(max(shape[:, 0])), int(max(shape[:, 1]))]
        #print 'Bbox:{}, r:{}'.format(bbox, float(bbox[2]-bbox[0])/(bbox[3]-bbox[1]))

        r = float(bbox[2]-bbox[0])/(bbox[3]-bbox[1])

        #print 'r={}'.format(r)
        if r < 1:
            update = [-(1-r)*(bbox[2]-bbox[0])/2, 0,
                      (1-r)*(bbox[2]-bbox[0])/2, 0]
        else:
            update = [0, -(r-1)*(bbox[3]-bbox[1])/2, 0,
                      (r-1)*(bbox[3]-bbox[1])/2]

        sbbox = np.asarray(bbox, dtype=np.int16) + \
            np.asarray(update, dtype=np.int16)
        #print 'Sbbox:{}, r:{}'.format(sbbox, float(sbbox[2]-sbbox[0])/(sbbox[3]-sbbox[1]))
        #print 'Sbbox2:{}'.format(np.vstack([[sbbox[1], sbbox[0]], [sbbox[3], sbbox[0]], [sbbox[1], sbbox[2]], [sbbox[3], sbbox[2]]]))
        # return np.vstack([[sbbox[2], sbbox[0]], [sbbox[1], sbbox[0]], [sbbox[3], sbbox[1]], [sbbox[3], sbbox[2]]])
        return np.vstack([[sbbox[1], sbbox[0]], [sbbox[3], sbbox[0]], [sbbox[1], sbbox[2]], [sbbox[3], sbbox[2]]])
    else:
        return np.zeros((4, 2))


def extract_face(i, im, ext=1.1, sz=224, verbose=False):
    # Extract face from image
    face_detector = dlib.get_frontal_face_detector()
    dets = face_detector(im, 1)

    if dets:
        rect = [d for d in dets][0]
        geom = np.asarray([[rect.top(), rect.left(), ], [rect.bottom(), rect.left()],
                           [rect.top(), rect.right(), ], [rect.bottom(), rect.right()]])

        face, _, _ = extract(im, geom, extension=ext, size=sz)

        if verbose and i % 20 == 0:
            print '         Extracting face {}'.format(i)

        detected, face = (True, face)
    else:
        if verbose:
            print '         No face detected'

        detected, face = (False, np.zeros((sz, sz)))

    return detected, face


def robust_extract_face(im, model, ext=1.1, sz=224, verbose=False):
    # Extract face from image
    cnn_face_detector = dlib.cnn_face_detection_model_v1(model)

    dets = cnn_face_detector(im, 1)

    for i, d in enumerate(dets):
        geom = np.asarray([[d.rect.top(), d.rect.left(), ], [d.rect.bottom(), d.rect.left()],
                           [d.rect.top(), d.rect.right(), ], [d.rect.bottom(), d.rect.right()]])

        face, _, _ = extract(im, geom, extension=ext, size=sz)
        detected, face = (True, face)
    else:
        if verbose:
            print '\t\t\tNo face detected'

        detected, face = (False, np.zeros((sz, sz)))

    return detected, face.astype(dtype=np.uint8)
