from processor.aligner import batch_align
from reader.dataset import ReaderCKplus
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import scipy.io as io
import getopt
import frontalizer.check_resources as check
import dlib
from frontalizer.frontalize import ThreeD_Model
from  scipy.ndimage import imread
from  scipy.misc import imresize

def run_femo_align():
    '''
    opts, args = getopt.getopt(argv, '')
    (ipath, opath, mpath, start, stop) = \
        (
            args[0], args[1], args[2], int(args[3]), int(args[4])
        )
    '''

    (ipath, opath, mpath, start, stop) = ('/Users/cipriancorneanu/Research/data/fake/extracted_faces/',
                             '/Users/cipriancorneanu/Research/data/fake/aligned_faces/',
                             '/Users/cipriancorneanu/Research/code/facepp/models/', 1 , 2)


    # Load models
    check.check_dlib_landmark_weights(mpath + 'shape_predictor_models')
    predictor = dlib.shape_predictor(mpath + 'shape_predictor_models/shape_predictor_68_face_landmarks.dat')
    model3D = ThreeD_Model(mpath + 'frontalization_models/model3Ddlib.mat', 'model_dlib')
    eyemask = np.asarray(io.loadmat(mpath + 'frontalization_models/eyemask.mat')['eyemask'])

    for person in range(start, stop):
        for emo in range(0,12):
            fname = 'femo_extracted_faces_' + str(person) + '_' + str(emo) + '.pkl'

            if os.path.exists(ipath+fname):
                print 'Aligning batch {}'.format(fname)
                faces =  cPickle.load(open(ipath+fname, 'rb'))
                afaces, ageoms = batch_align(faces, model3D, eyemask, predictor)

                cPickle.dump(
                    {'faces':afaces, 'geoms': ageoms},
                    open(opath+'femo_aligned_faces_' + str(person) + '_' + str(emo)+'.pkl', 'wb'),
                    cPickle.HIGHEST_PROTOCOL
                )

def read_femo_extracted_faces(argv):
    opts, args = getopt.getopt(argv, '')
    (ipath, opath) = \
        (
            args[0], args[1]
        )

    n_persons, n_classes = (54, 12)
    person_keys = ['_'+str(x+1)+'_' for x in np.arange(0,n_persons)]
    target_keys = ['act_HAPPY', 'act_SAD', 'act_CONTEMPT', 'act_SURPRISED', 'act_DISGUST', 'act_ANGRY',
                   'fake_HAPPY', 'fake_SAD', 'fake_CONTEMPT', 'fake_SURPRISED', 'fake_DISGUST', 'fake_ANGRY']

    data = [[None for _ in range(n_classes)] for _ in range(n_persons)]

    for p_key in range(1,n_persons+1):
        for t_key in  [f for f in os.listdir(ipath+str(p_key)+'/') if not f.startswith('.')]:
            print 'person:{} target:{}'.format(str(p_key),t_key)

            files = [f for f in os.listdir(ipath+str(p_key)+'/'+t_key+'/') if f.endswith('.png')]
            files.sort()

            ims = np.zeros((len(files),224,224,3), dtype=np.uint8)
            for i,f in enumerate(files):
                im = imread(ipath+str(p_key)+'/'+t_key+'/'+f)
                ims[i,:,:] = np.asarray(imresize(im,(224, 224)), dtype=np.uint8)

            print('Dumping data to ' + opath)
            cat, emo = t_key.split('_')
            cPickle.dump(
                ims, open(opath+'femo_extracted_faces_'+str(p_key)+'_'+str(_map_class(cat, emo))+'.pkl', 'wb'),
                cPickle.HIGHEST_PROTOCOL
            )

def _map_class(category, emo):
    map= {'act':{'HAPPY':0, 'SAD':1, 'CONTEMPT':2, 'SURPRISED':3, 'DISGUST':4, 'ANGRY':5},
          'fake':{'HAPPY':6, 'SAD':7, 'CONTEMPT':8, 'SURPRISED':9, 'DISGUST':10, 'ANGRY':11}}

    return map[category][emo]


def run_ckp_align():
    '''
    opts, args = getopt.getopt(argv, '')
    (ipath, opath) = (args[0], args[1])
    '''

    (ipath, opath, mpath) = ('/Users/cipriancorneanu/Research/data/ck/',
                             '/Users/cipriancorneanu/Research/data/ck/',
                             '/Users/cipriancorneanu/Research/code/facepp/models/')

    # Read data
    r_ckp = ReaderCKplus(ipath)
    dt = r_ckp.read('ckp.pkl')
    faces = dt['images']

    # Load models
    check.check_dlib_landmark_weights(mpath + 'shape_predictor_models')
    predictor = dlib.shape_predictor(mpath + 'shape_predictor_models/shape_predictor_68_face_landmarks.dat')
    model3D = ThreeD_Model(mpath + 'frontalization_models/model3Ddlib.mat', 'model_dlib')
    eyemask = np.asarray(io.loadmat(mpath + 'frontalization_models/eyemask.mat')['eyemask'])

    # Align
    afaces, ageoms = ([], [])
    for sequence in faces:
        aface, ageom = batch_align(sequence, model3D, eyemask, predictor)
        afaces.append(aface)
        ageoms.append(ageom)

    '''
    for s, (seqf, seqg) in enumerate(zip(afaces, ageoms)):
        for i in range(0, len(seqf), 40):
            plt.imshow(seqf[i, ...])
            plt.scatter(seqg[i, :, 0], seqg[i, :, 1])
            plt.savefig('/Users/cipriancorneanu/Research/data/fake/aligned_faces/final_'+str(s) + '_' + str(i))
            plt.clf()
    '''

    # Save data
    cPickle.dump(
        {'faces':afaces, 'geoms': ageoms, 'emos': dt['emos'], 'subjects': dt['subjects'], 'sequences': dt['sequences']},
        open(opath+'a_ckp.pkl', 'wb'),
        cPickle.HIGHEST_PROTOCOL
    )

if __name__ == "__main__":
    '''
    print '####### Extract faces ######'
    read_femo_extracted_faces(sys.argv[1:])
    '''
    '''
    print '####### Align faces ######'
    run_femo_align()
    '''

    dt = cPickle.load(open('/Users/cipriancorneanu/Research/data/fake/aligned_faces/femo_aligned_faces_1_2.pkl'))
    faces, geoms = dt['faces'], dt['geoms']

    for i in range(0, 10):
        plt.imshow(faces[i, ...])
        plt.scatter(geoms[i, :, 0], geoms[i, :, 1])
        plt.savefig('/Users/cipriancorneanu/Research/data/fake/aligned_faces/final_'+ str(i))
        plt.clf()
