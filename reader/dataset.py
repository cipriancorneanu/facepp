__author__ = 'cipriancorneanu'

import re
from facepp.processor.aligner import align
import matplotlib.pyplot as plt
import cPickle
import scipy.io as io
import getopt
from facepp.frontalizer.check_resources import check_dlib_landmark_weights
import dlib
from facepp.frontalizer.frontalize import ThreeD_Model
from facepp.processor.encoder import encode_parametric
from facepp.processor.partitioner import slice, deconcat
from extractor import extract, extract_face
from reader import *
import time
from joblib import Parallel, delayed
import multiprocessing

class ReaderFera2017():
    def __init__(self, path):
        self.path = path
        self.path_ims = path + 'ims/'
        self.path_occ = path + 'occ/'
        self.path_int = path + 'int/'
        self.subjects = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009',
                         'F010', 'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019',
                         'F020', 'F021', 'F022', 'F023'
                         'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009',
                         'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']
        self.tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
        self.poses = [str(i) for i in range(1,10)]
        self.aus = [1, 4, 6, 7, 10, 12, 14, 15, 17, 23]
        self.aus_int = ['AU01', 'AU04', 'AU06', 'AU10', 'AU12', 'AU14', 'AU17']

    def read(self, opath, mpath, poses=[6], cores=4):
        print 'List of selected subjects: {}'.format(self.subjects)

        # Load models
        check_dlib_landmark_weights(mpath + 'shape_predictor_models')
        predictor = dlib.shape_predictor(mpath + 'shape_predictor_models/shape_predictor_68_face_landmarks.dat')
        model3D = ThreeD_Model(mpath + 'frontalization_models/model3Ddlib.mat', 'model_dlib')
        eyemask = np.asarray(io.loadmat(mpath + 'frontalization_models/eyemask.mat')['eyemask'])

        # Get list of subjects
        for subject in self.subjects:
            for task in self.tasks:
                for pose in poses:
                    start_time = time.time()
                    vidname = self.path_ims + 'FERA17_TR_' + subject + '_' + task + '_' + str(pose) + '.mp4'
                    occname = self.path_occ + 'FERA17_TR_' + subject + '_' + task + '.csv'

                    # Read video
                    if os.path.exists(vidname) and os.path.exists(occname):
                        dt = {'ims': [], 'geoms': [], 'occ':[], 'int':[], 'subjects':[], 'tasks':[], 'poses':[]}
                        start_time = time.time()
                        ims = read_video(vidname, colorspace='L')
                        print 'Reading video file {} in {}s'.format(vidname, (time.time() - start_time))

                        occ = read_csv(occname)
                        int = self._read_au_intensities(subject, task)

                        print '     Extract faces and resize '
                        faces = Parallel(n_jobs=cores)(delayed(extract_face)(i,im) for i,im in enumerate(ims))

                        print '     Align faces'
                        aligned = [align(i, face, model3D, eyemask, predictor) if face_detected
                                   else (np.zeros((face.shape[0], face.shape[1], 3)), np.zeros((68,2)))
                                   for i,(face_detected,face) in enumerate(faces)]

                        '''
                        aligned = Parallel(n_jobs=2)(delayed(align)(i, face, model3D, eyemask, predictor)
                                                                    for i,face in enumerate(faces)
                        '''

                        afaces, ageoms = (np.asarray([x[0] for x in aligned], dtype=np.uint8),
                                          np.asarray([x[1] for x in aligned], dtype=np.float16))

                        # Save
                        dt['ims'].append(afaces)
                        dt['occ'].append([x[1:] for x in occ[1:]])
                        dt['int'].append(int)
                        dt['geoms'].append(ageoms)
                        dt['subjects'].append(subject)
                        dt['tasks'].append(task)
                        dt['poses'].append(str(pose))

                        cPickle.dump(dt, open(opath+'fera17_'+ subject + '_' + task + '_' + str(pose) + '.pkl', 'wb'),
                                     cPickle.HIGHEST_PROTOCOL)
                        print '     Total time per video: {}'.format(time.time() - start_time)

    def read_geom(self, path, fname):
        if os.path.exists(path + fname):
            return cPickle.load(open(path+fname, 'rb'))

        dt = {'geoms': [], 'occ':[], 'int':[], 'subjects':[], 'tasks':[], 'poses':[]}

        # If file does not exist load from original data
        for subject in self.subjects:
            for task in self.tasks:
                for pose in self.poses[5:6]:
                    fname_orig = path + 'fera17_' + subject + '_' + task + '_' + pose + '.pkl'

                    sequence = cPickle.load(open(fname_orig, 'rb'))

                    dt['occ'].append(sequence['occ'][0])
                    dt['int'].append(sequence['int'][0])
                    dt['geoms'].append(sequence['geoms'][0])
                    dt['subjects'].append(sequence['subjects'])
                    dt['tasks'].append(sequence['tasks'])
                    dt['poses'].append(sequence['poses'])

        # Encode geometry
        slices = slice(dt['geoms'])
        geom = np.squeeze(np.concatenate([[x for x in item ] for item in  dt['geoms']]))
        enc_geom, _, _ = encode_parametric(np.asarray(geom, dtype=np.float32))
        enc_geom = deconcat(enc_geom, slices)
        dt['geoms'] = enc_geom

        cPickle.dump(dt, open(path+fname, 'wb'), cPickle.HIGHEST_PROTOCOL)

        return dt

    def read_sift(self):
        pass


    def _read_au_intensities(self, subject, task):
        dt = []
        for au in self.aus_int:
            fname = self.path_int + 'FERA17_TR_' + subject + '_' + task + '_' + au + '_Int.csv'
            dt.append([x[1] for x in read_csv(fname)])

        return np.asarray(dt)

    def _get_subjects(self):
        fnames = [f for f in os.listdir(self.path_ims) if f.endswith('.mp4')]
        return list(set([re.split('_', f)[2] for f in fnames]))

class ReaderCKplus():
    def __init__(self, path):
        self.path = path
        self.path_im = path + 'cohn-kanade-images/'
        self.path_lm = path + 'Landmarks/'
        self.path_emo = path + 'Emotion/'

    def read(self, fname):
        if os.path.exists(self.path+fname):
            return cPickle.load(open(self.path+fname, 'rb'))

        dt = {'images':[], 'landmarks':[], 'emos':[], 'subjects':[], 'sequences':[]}

        # Get directory structure
        subjects = sorted([f for f in os.listdir(self.path_im)])
        sequences = [[x for x  in sorted(os.listdir(self.path_im+sub+'/')) if not x[0]=='.'] for sub in subjects]

        print('###### READING CK+ DATASET ######')
        for subject, subject_sequences in zip(subjects, sequences):
            for sequence in subject_sequences:
                print('Subject:{}, Sequence:{}'.format(subject,sequence))
                rpath = subject+'/'+sequence + '/'

                im_seq = read_folder(self.path_im+rpath)

                # Read and save if all corresponding info is available
                if os.path.exists(self.path_lm+rpath) and os.path.exists(self.path_emo+rpath):
                    lm_seq = np.asarray(read_folder(self.path_lm+rpath), dtype=np.float16)[:,:,::-1]

                    # Extract face from landmarks and resize
                    S = map(list, zip(*[extract(i,l,1.05,224) for i,l in zip(im_seq, lm_seq)]))

                    dt['images'].append(np.asarray(S[0], dtype=np.uint8))
                    dt['landmarks'].append(np.asarray(S[1], dtype=np.float16))
                    dt['emos'].append(read_folder(self.path_emo+rpath))
                    dt['subjects'].append(subject)
                    dt['sequences'].append(sequence)
                else:
                    print 'Skip: Not all corresponding labels exists.'

        cPickle.dump(dt, open(self.path+fname, 'wb'), cPickle.HIGHEST_PROTOCOL)

        return dt

class Reader300vw():
    def __init__(self, path):
        self.path = path
        self.partitions = dict({'train0':['002', '003', '004', '007', '009', '010', '011', '013', '015', '017', '018', '019', '020', '022', '025', '027', '028', '029', '033', '034'],
                           'train1':['035', '037', '039', '041', '043', '044', '046', '048', '049', '053', '057', '059', '112', '113', '114', '115', '120', '123', '124', '125'],
                           'train2':['126', '138', '143', '144', '150', '160', '203', '204', '205', '208', '211', '212', '213', '214', '223', '224', '225', '401', '402', '403', '404'],
                           'train3':['405', '406', '408', '409', '410', '411', '412', '505', '506', '507', '508', '510', '511', '514', '515', '516', '517', '518', '519', '520'],
                           'train4':['522', '524', '525', '526', '528', '529', '530', '531', '533', '538', '540', '541', '546', '547', '548', '550', '551', '553', '558', '559', '562'],
                           'test':['001', '016', '031', '047', '119', '158', '218', '407', '509', '521', '537', '557']})

    def read(self):
        print('###### READING 300VW DATASET ######')
        for p in self.partitions:
            if os.path.exists(self.path+p+'.pkl'):
                return cPickle.load(open(self.path+p+'.pkl', 'rb'))

            print ("Loading " + p)
            data = self._read_partition(self.partitions[p])
            cPickle.dump(data, open(self.path+p+'.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

            return data

    def _read_partition(self, partition):
        dt = {'images':[], 'landmarks':[], 'T':[], 'names': []}

        for d in partition[:2]:
            print ('         loading sequence ' + str(d))

            im_seq = read_folder(self.path + d + '/')
            lm_seq = np.asarray(read_folder(self.path + d + '/annot/'), dtype=np.float16)[:,:,::-1]# !filter3:-1

            S = map(list, zip(*[extract(i,l) for i,l in zip(im_seq[0], lm_seq)]))

            # Guard data
            dt['images'].append(S[0])
            dt['landmarks'].append(S[1])
            dt['T'].append(S[2])
            dt['names'].append(d)
        return dt

class ReaderPain():
    def __init__(self, path):
        self.path = path
        self.path_im = path + 'Images/'
        self.path_lm = path + 'AAM_landmarks/'
        self.path_au = path + 'Frame_Labels/FACS/'
        self.au_list = [4,6,7,9,10,12,20,25,26,43]

    def read(self, fname):
        if os.path.exists(self.path+fname):
            return cPickle.load(open(self.path+fname, 'rb'))

        dt = {'images':[], 'landmarks':[], 'aus':[], 'subjects':[], 'sequences':[]}

        # Get directory structure
        subjects = sorted([f for f in os.listdir(self.path_im)])
        sequences = [sorted(os.listdir(self.path_im+sub+'/')) for sub in subjects]

        print('###### READING PAIN DATASET ######')
        for subject, subject_sequences in zip(subjects, sequences):
            for sequence in subject_sequences:
                print('Subject:{}, Sequence:{}'.format(subject,sequence))
                rpath = subject+'/'+sequence + '/'

                im_seq = read_folder(self.path_im+rpath)
                lm_seq = np.asarray(read_folder(self.path_lm+rpath), dtype=np.float16)[:,:,::-1]

                # Extract face and resize
                S = map(list, zip(*[extract(i,l,1,30) for i,l in zip(im_seq, lm_seq)]))

                dt['images'].append(S[0])
                dt['landmarks'].append(S[1])
                dt['aus'].append(read_folder(self.path_au+rpath))
                dt['subject'].append(subject)
                dt['sequence'].append(sequence)

        cPickle.dump(dt, open(self.path+fname, 'wb'), cPickle.HIGHEST_PROTOCOL)

        return dt

    def _vectorize_au(self, au):
        ''' Pass sample from text format to vectorial format'''
        au_vector = np.zeros(len(self.au_list), dtype=np.uint8)

        # Get info
        au = int(float(au[0]))
        intensity = int(float(au[1]))

        # Vectorize
        if au in self.au_list:
            au_vector[self.au_list.index(au)] = intensity
            return au_vector
        else:
            return None

    def _vectorize_au_sequence(self, au_seq):
        return [self._vectorize_au(au) for frame in au_seq for au in frame]

class ReaderDisfa():
    def __init__(self, path):
        self.path = path
        self.path_lm = path + 'Landmark_Points/'
        self.path_au = path + 'ActionUnit_Labels/'
        self.path_vidl = path + 'Video_LeftCamera/'
        self.path_vidr = path + 'Video_RightCamera/'

    def read(self, fname):
        if os.path.exists(self.path+fname):
            return cPickle.load(open(self.path+fname, 'rb'))

        dt = {'images':[], 'landmarks':[], 'aus':[], 'subjects':[]}

        subjects = sorted([f for f in os.listdir(self.path_au)])

        print('###### READING DISFA DATASET ######')

        for subject in subjects:
            print('Reading subject ' + subject)

            lm_seq = np.asarray([np.fliplr(x['pts'])
                                 for x in read_folder(self.path_lm+subject+'/'+'frame_lm/', self._lm_file_sorter)])
            au_seq = read_folder(self.path_au+subject+'/', self._au_file_sorter)
            im_seq = read_video(self.path_vidl+'LeftVideo'+subject+'_comp.avi')

            # Extract face and resize
            S = map(list, zip(*[extract(i,l) for i,l in zip(im_seq, lm_seq)]))

            dt['images'].append(S[0])
            dt['landmarks'].append(S[1])
            dt['aus'].append(self._vectorize_au_sequence(au_seq))
            dt['subject'].append(subject)

        cPickle.dump(dt, open(self.path+fname, 'wb'), cPickle.HIGHEST_PROTOCOL)
        return dt

    def _vectorize_au_sequence(self, au_seq):
        return np.asarray(np.transpose(np.vstack([ [int(x[0].split(',')[1]) for x in au] for au in au_seq])),
                          dtype = np.uint8)

    def _au_file_sorter(self, files):
        # Sort files according to integer key in file name
        sorting = list(np.argsort([int(f.split('au')[1].split('.')[0]) for f in files]))
        return  [files[s] for s in sorting]

    def _lm_file_sorter(self, files):
        return sorted(files)

if __name__ == '__main__':
    path_300vw = '/Users/cipriancorneanu/Research/data/300vw/'
    path_pain = '/Users/cipriancorneanu/Research/data/pain/'
    path_disfa = '/Users/cipriancorneanu/Research/data/disfa/'
    path_server_pain = '/home/corneanu/data/pain/'
    path_server_disfa = '/home/corneanu/data/disfa/'
    path_ckplus = '/Users/cipriancorneanu/Research/data/ck/'
    path_fera2017 = '/Users/cipriancorneanu/Research/data/fera2017/'
    path_fera2017_server = '/data/hupba2/corneanu/data/fera2017/'
    path_align_models = '/Users/cipriancorneanu/Research/code/facepp/models/'

    fera = ReaderFera2017(path_fera2017_server)
    dt = fera.read_geom(path_fera2017 + 'aligned/', 'fera2017_geom.pkl')

    '''
    ims, geoms = (dt['ims'][0], dt['geoms'][0])

    for i, (im, geom) in enumerate(zip(ims, geoms)):
        plt.imshow(im)
        plt.scatter(geom[:, 0], geom[:, 1])
        plt.savefig('/Users/cipriancorneanu/Research/data/fera2017/results/' + str(i) + '.png')
        plt.clf()
    '''