__author__ = 'cipriancorneanu'

import os
import scipy.io
import numpy as np
import imageio
import skimage.color
import csv

def read_folder(path, sorter=None):
    # Get files from folder
    files, extensions = get_files(path, sorter)

    #Read according to extension
    return [read_txt(path+file) if ext=='txt' else
            read_video(path+file) if ext=='avi' else
            read_mat(path+file) if ext=='mat' else
            read_image(path+file, colorspace='RGB') for ext, file in zip(extensions, files)]

def get_files(path, sorter=None):
    # Get files from path by filtering hidden and directories
    files = [f for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))
                    and not f.startswith('.')]

    if sorter: files = sorter(files)
    extensions = [f.split('.')[-1] for f in files]

    return (files, extensions)

def read_mat(fname):
    return scipy.io.loadmat(fname)

def read_video(fname, colorspace='RGB'):
    vid = imageio.get_reader(fname, 'ffmpeg')

    # For the moment just read some frames to speedup
    frames = np.asarray([np.asarray(vid.get_data(i)) for i in range(0, len(vid))])

    if colorspace=='L':
        return np.asarray([np.asarray(255*skimage.color.rgb2gray(frame), dtype=np.uint8) for frame in frames])
    elif colorspace == 'RGB':
        return frames

def read_image(fname, colorspace='L'):
    return np.asarray(scipy.misc.imread(fname, mode=colorspace), dtype=np.uint8)

def read_txt(fname, start=0, stop=None):
    # Read lines from text file from start to stop
    with open(fname) as f:
        return [line.split() for line in f][start:stop]

def read_csv(fname):
    with open(fname, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        return [[int(x) for x in row[0].split(',')] for row in spamreader ]

if __name__ == '__main__':
    #read_csv('/Users/cipriancorneanu/Research/data/fera2017/train/occ/FERA17_TR_F003_T1.csv')
    '''
    path = '/Users/cipriancorneanu/Research/data/fake/aligned_faces/femo_aligned_faces_12_9.pkl'

    import cPickle
    import matplotlib.pyplot as plt

    dt = cPickle.load(open(path, 'rb'))

    for i, (im, geom) in enumerate(zip(dt['faces'], dt['geoms'])):
        if i%20 == 0:
            plt.imshow(im)
            plt.scatter(geom[:,0], geom[:,1])
            plt.savefig('/Users/cipriancorneanu/Research/data/fake/' + str(i) + '.png')
            plt.clf()
    '''
