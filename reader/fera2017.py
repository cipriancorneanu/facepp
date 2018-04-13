__author__ = 'cipriancorneanu'

import os
import time
from reader import read_video, read_csv
from extractor import robust_extract_face
import numpy as np
import h5py
import skimage.transform

import scipy.misc
path_face_extractor = '/home/chip/code/dlib/python_examples/mmod_human_face_detector.dat'


def prepare(ipath, opath, subject, pose, ofname='out.h5'):

    # Get all video files from input path
    video_files = sorted([f for f in os.listdir(
        ipath+'/ims') if subject in f and str(pose)+'.' in f])
    print video_files

    for vidname in video_files:
        (dataset, partition, subject, task,
         pose) = vidname.split('.')[0].split('_')
        occname = ipath+'occ/' + dataset + '_' + \
            partition + '_' + subject + '_' + task + '.csv'

        # Read data
        if os.path.exists(ipath+'ims/'+vidname):
            start_time = time.time()
            ims = read_video(ipath+'ims/'+vidname, colorspace='RGB')
            print('Reading video file {} in {}s'.format(
                vidname, (time.time() - start_time)))

            aus = np.asarray(read_csv(occname))[1:, 1:]
            print('\tExtract faces and resize ')
            faces, valids = [], []
            for i, im in enumerate(ims):
                start_time = time.time()
                im = skimage.transform.resize(im, (im.shape[0] / 2, im.shape[1] / 2),
                                              preserve_range=True).astype(np.uint8)

                valid, extracted_face = robust_extract_face(
                    i, im, model=path_face_extractor, ext=1.5, sz=224, verbose=False)

                '''print('extracted face : {}'.format(extracted_face.shape))'''

                faces.append(extracted_face)
                valids.append(valid)

                if i % 50 == 0:
                    print('\t\t extracting face {}/{} in {}s'.format(
                        i, len(ims), time.time()-start_time))

            aus = aus[valids]
            faces = np.stack(np.asarray(faces)[valids], axis=0)

            print('\tDetected {} from {} faces'.format(
                faces.shape[0], len(ims)))

            '''
            print('Print some samples for testing')
            for i in range(0, faces.shape[0], 10):
                print('{}'.format(aus[i]))
                scipy.misc.toimage(
                    faces[i], cmin=0, cmax=255).save(str(i)+'.jpg')
            '''

            # Dump data
            if os.path.isfile(opath+ofname):
                file = h5py.File(opath+ofname, 'r+')
            else:
                file = h5py.File(opath+ofname, 'w')

            print (partition+'/'+subject+'/'+task+'/'+pose)

            segment = file.create_group(
                partition+'/'+subject+'/'+task+'/'+pose)
            segment.create_dataset('faces', data=faces)
            segment.create_dataset('aus', data=aus)


if __name__ == '__main__':
    '''['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                       'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020',
                       'F021', 'F022', 'F023', 'M001', 'M002', 'M003', 'M004']'''
    subjects_train = ['M005', 'M006', 'M007',
                      'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']

    subjects_validation = ['F007', 'F008', 'F009', 'F010', 'F011', 'M001', 'M002', 'M003', 'M004',
                           'M005', 'M006', 'rF001', 'rF002', 'rM001', 'rM002', 'rM003', 'rM004',
                           'rM005', 'rM006', 'rM007']

    for subject in subjects_train:
        for pose in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            prepare('/data/data1/datasets/fera2017/train/',
                    '/data/data1/datasets/fera2017/', subject, pose, ofname='fera_new_test.h5')
