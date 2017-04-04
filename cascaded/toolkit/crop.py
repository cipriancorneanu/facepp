__author__ = 'cipriancorneanu'

import scipy
import numpy as np


def crop_resize(im, lm, ext=1.3):
    im, lm, t1 = crop(im, lm, ext)
    im, lm, t2 = thumbnail(im, lm, 200)
    return im, lm, np.dot(t2, t1)

def crop(im, lm, ext=1.3):
    '''
    Crop bbox around landmarks from image
    :param im: input image to crop
    :param lm: a set of landmarks
    :param ext: bbox extension in %
    :return: cropped image, cropped landmarks
    '''

    def extend_rect(rect, ext):
        # Compute extension params
        h2, w2 = int(ext*(rect[2]-rect[0])) / 2, int(ext*(rect[3]-rect[1])) / 2
        center_x, center_y = (rect[2]+rect[0])/2, (rect[3]+rect[1])/2

        # Extend bbox
        return [center_x-h2, center_y-w2, center_x+h2, center_y+w2]

    # Get bbox
    bbox = [int(min(lm[:, 0])), int(min(lm[:, 1])), int(max(lm[:, 0])), int(max(lm[:, 1]))]

    # Extend bbox
    bbox = extend_rect(bbox, ext)

    # Bbox inside image
    ibbox = [int(max(0, bbox[0])), int(max(0, bbox[1])), min(im.shape[0], bbox[2]), min(im.shape[1], bbox[3])]

    # If inside get image if out put to zero
    imc = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1]), dtype=np.uint8)
    imc[0:ibbox[2]-ibbox[0], 0:ibbox[3]-ibbox[1]] = im[ibbox[0]:ibbox[2], ibbox[1]:ibbox[3]]

    return (
        imc,
        lm-bbox[:2],
        np.array([[1, 0, 0], [0, 1, 0], [bbox[0], bbox[1], 1]])
    )

def thumbnail(im, lm, sz):
    '''
    Resize image by keeping aspect ratio
    :param im: image to resize
    :param size: new size. Thumbnail to rectangular sizes only.
    :return: thumbnail image, transformation btw. input and output landmarks
    '''
    # Get the dominant dimension

    def pad(mat, shape):
        # Pad values to mat in order to get shape
        padded = np.zeros(shape, dtype=mat.dtype)
        padded[0:mat.shape[0], 0:mat.shape[1]] = mat
        return padded

    # Get dominant dimension
    ddim = im.shape.index(max(im.shape))

    # Compute ratio and resize
    r = float(sz)/im.shape[ddim]
    im = scipy.misc.imresize(im, r)
    lm = lm*r

    # Thumbnail on smaller dimension
    margins = (np.array([sz,sz]) - np.array(im.shape) ) / 2
    imt = np.zeros((sz, sz), dtype=np.uint8)
    imt[margins[0]:margins[0]+im.shape[0],margins[1]:margins[1]+im.shape[1]] = im
    lm += margins

    return imt, lm, np.array([[1 / r, 0, 0], [0, 1 / r, 0], [-margins[0] / r, -margins[1] / r, 1]])
