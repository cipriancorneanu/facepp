__author__ = 'cipriancorneanu'

import os, time
from reader import read_video, read_csv
from extractor import extract_face
import numpy as np
import h5py

def prepare(ipath, opath, ofname='fera2017.h5'):

    # Get all video files from input path
    video_files = sorted([f for f in os.listdir(ipath+'/ims')])

    # Parse video files
    info =  [f.split('.')[0].split('_') for f in video_files]

    for vidname in video_files:
        (dataset, partition, subject, task, pose)  = vidname.split('.')[0].split('_')
        occname = ipath+'occ/' + dataset + '_' + partition + '_' + subject + '_' + task + '.csv'

        # Read data
        if os.path.exists(ipath+'ims/'+vidname):
            start_time = time.time()
            ims = read_video(ipath+'ims/'+vidname, colorspace='RGB')
            print('Reading video file {} in {}s'.format(vidname, (time.time() - start_time)))

            aus = read_csv(occname)
            print('     Extract faces and resize ')
            faces = np.asarray([extract_face(i,im, ext=1.5, sz=112, verbose=True) for i,im in enumerate(ims)],
                               dtype=np.uint8)

            # Dump data
            if os.path.isfile(opath+ofname):
                file = h5py.File(opath+ofname, 'r+')
            else:
                file = h5py.File(opath+ofname, 'w')

            segment = file.create_group(partition+'/'+subject+'/'+task+'/'+pose)
            segment.create_dataset('faces', data=faces)
            segment.create_dataset('aus', data=aus)