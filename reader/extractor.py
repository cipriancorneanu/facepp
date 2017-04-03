__author__ = 'cipriancorneanu'

import numpy as np
import scipy.misc
from reader import *

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
    imc = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1]), dtype=np.uint8)
    imc[0:ibbox[2]-ibbox[0], 0:ibbox[3]-ibbox[1]] = im[ibbox[0]:ibbox[2], ibbox[1]:ibbox[3]]

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
    margins = (np.array([size,size]) - np.array(im.shape) ) / 2
    imt = np.zeros((size, size), dtype=np.uint8)
    imt[margins[0]:margins[0]+im.shape[0],margins[1]:margins[1]+im.shape[1]] = im

    return (
        imt,
        shape + margins,
        np.array([[1 / r, 0, 0], [0, 1 / r, 0], [-margins[0] / r, -margins[1] / r, 1]])
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
        padded = np.zeros(shape, dtype=mat.dtype)
        padded[0:im.shape[0], 0:im.shape[1]] = im
        return padded

if __name__ == '__main__':

    path = '/Users/cipriancorneanu/Research/data/300vw/'

    dirs = ['002']
    dt = {'images':[], 'landmarks':[], 'T':[], 'names': []}

    for d in dirs:
        print '         loading sequence ' + str(d)

        im_seq = read_folder(path + d + '/')
        lm_seq = np.asarray(read_folder(path + d + '/annot/'), dtype=np.float16) #! Cutting text reading

        S = map(list, zip(*[extract(i,l) for i,l in zip(im_seq[0], lm_seq)]))

        # Check if transformation is correct
        Lprim = [np.dot(np.hstack((L,np.ones((68,1)))), T) for L,T in zip(S[1], S[2])]
        D = [L - L_[:,:2] for L, L_ in zip(lm_seq, Lprim)]

        # Guard data
        dt['images'].append(S[0])
        dt['landmarks'].append(S[1])
        dt['T'].append(S[2])
        dt['names'].append(d)