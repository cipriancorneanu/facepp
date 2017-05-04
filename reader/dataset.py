__author__ = 'cipriancorneanu'

import re
from facepp.processor.aligner import align
import cPickle
import scipy.io as io
from facepp.frontalizer.check_resources import check_dlib_landmark_weights
import dlib
from facepp.frontalizer.frontalize import ThreeD_Model
from facepp.processor.encoder import encode_parametric
from facepp.processor import partitioner
from extractor import extract, extract_face
from reader import *
import time
from joblib import Parallel, delayed

class ReaderFera2017():
    def __init__(self, path):
        self.path = path
        self.path_ims = path + 'ims/'
        self.path_occ = path + 'occ/'
        self.path_int = path + 'int/'
        self.poses = [str(i) for i in range(1,10)]
        self.aus = [1, 4, 6, 7, 10, 12, 14, 15, 17, 23]
        self.aus_int = ['AU01', 'AU04', 'AU06', 'AU10', 'AU12', 'AU14', 'AU17']

    def read(self, opath, mpath, partition = 'train', poses=[6], cores=4):
        if partition == 'train':
            root = 'FERA17_TR_'
        elif partition == 'validation':
            root = 'FERA17_VA_'

        subjects = self._get_subjects()
        print 'List of selected subjects: {}'.format(subjects)

        # Load models
        check_dlib_landmark_weights(mpath + 'shape_predictor_models')
        predictor = dlib.shape_predictor(mpath + 'shape_predictor_models/shape_predictor_68_face_landmarks.dat')
        model3D = ThreeD_Model(mpath + 'frontalization_models/model3Ddlib.mat', 'model_dlib')
        eyemask = np.asarray(io.loadmat(mpath + 'frontalization_models/eyemask.mat')['eyemask'])

        # Get list of subjects
        for subject in subjects:
            print 'List of selected tasks for subject {}: {}'.format(subject, self._get_tasks(subject))
            for task in self._get_tasks(subject):
                for pose in poses:
                    start_time = time.time()
                    vidname = self.path_ims + root + subject + '_' + task + '_' + str(pose) + '.mp4'
                    occname = self.path_occ + root + subject + '_' + task + '.csv'

                    # Read video
                    if os.path.exists(vidname) and os.path.exists(occname):
                        dt = {'ims': [], 'geoms': [], 'occ':[], 'int':[], 'subjects':[], 'tasks':[], 'poses':[]}
                        start_time = time.time()
                        ims = read_video(vidname, colorspace='L')
                        print 'Reading video file {} in {}s'.format(vidname, (time.time() - start_time))

                        occ = read_csv(occname)
                        int = self._read_au_intensities(root, subject, task)

                        print '     Extract faces and resize '
                        faces = Parallel(n_jobs=cores)(delayed(extract_face)(i,im) for i,im in enumerate(ims))

                        print '     Align faces'
                        aligned = [align(i, face, model3D, eyemask, predictor) if face_detected
                                   else (np.zeros((face.shape[0], face.shape[1], 3)), np.zeros((68,2)))
                                   for i,(face_detected,face) in enumerate(faces)]

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

        fnames = [f for f in os.listdir(self.path + '/aligned') if f.startswith('fera17') and f.endswith('.pkl')]
        dt = {'ims': [], 'geoms': [], 'occ':[], 'int':[], 'subjects':[], 'tasks':[], 'poses':[]}

        # If geom file does not exist load from original data
        for f in fnames:
            print 'Reading file {}'.format(f)

            seq = cPickle.load(open(self.path + '/aligned/' + f, 'rb'))

            ims, geoms, occ, int = self._filter_junk(np.asarray(seq['ims'][0], dtype=np.uint8),
                                                               np.asarray(seq['geoms'][0], dtype=np.float16),
                                                               np.asarray(seq['occ'][0], dtype=np.uint8),
                                                               np.transpose(np.asarray(seq['int'][0], dtype=np.uint8)))

            dt = self._accumulate_data(dt, ims = [], geoms = geoms, occ = occ,
                                        int = int, subjects = seq['subjects'],
                                        tasks = seq['tasks'], poses = seq['poses'])


        # Encode geometry
        geoms, slices = partitioner.concat(dt['geoms'])
        enc_geom, _, _ = encode_parametric(np.asarray(geoms, dtype=np.float32))
        dt['geoms'] = partitioner.deconcat(enc_geom, slices)

        # Dump encoded geometry
        cPickle.dump(dt, open(path+fname, 'wb'), cPickle.HIGHEST_PROTOCOL)

        return dt

    def read_batches(self, N):
        fnames = [f for f in os.listdir(self.path + '/aligned') if f.startswith('fera17') and f.endswith('.pkl')]

        # Load batches
        idxs = np.random.randint(0, N, len(fnames))
        print idxs

        for bat in range(0, N):
            print 'Loading batch {}'.format(bat)
            bat_fnames = [fnames[i] for i in np.where(idxs==bat)[0]]

            dt = {'ims': [],'geoms': [], 'occ':[], 'int':[], 'subjects':[], 'tasks':[], 'poses':[]}
            for f in bat_fnames:
                print '     Load sequence {}'.format(f)
                seq = cPickle.load(open(self.path + '/aligned/' + f, 'rb'))

                # Filter junk
                ims, geoms, occ, int = self._filter_junk(np.asarray(seq['ims'][0], dtype=np.uint8),
                                                               np.asarray(seq['geoms'][0], dtype=np.float16),
                                                               np.asarray(seq['occ'][0], dtype=np.uint8),
                                                               np.transpose(np.asarray(seq['int'][0], dtype=np.uint8)))

                # Accumulate data
                dt = self._accumulate_data(dt, ims, geoms, occ, int,
                                        subjects = [seq['subjects'][0] for i in range(0, len(seq['occ'][0]))],
                                        tasks = [seq['tasks'][0] for i in range(0, len(seq['occ'][0]))],
                                        poses = [seq['poses'][0] for i in range(0, len(seq['occ'][0]))])

            if len(dt['occ'])>0:
                # Vectorize and shuffle
                dt = self._shuffle_data(dt)

                # Dump
                print 'Dumping batch {}'.format(bat)
                cPickle.dump(dt, open(self.path+'/fera17_train_' + str(bat), 'wb'), cPickle.HIGHEST_PROTOCOL)
            else:
                print 'This batch is empty'

        return dt

    def _accumulate_data(self, dt, ims, geoms, occ, int, subjects, tasks, poses):
        dt['ims'].append(ims)
        dt['geoms'].append(geoms)
        dt['occ'].append(occ)
        dt['int'].append(int)
        dt['subjects'].append(subjects)
        dt['tasks'].append(tasks)
        dt['poses'].append(poses)

        return dt

    def _shuffle_data(self, dt):
        L = len(np.concatenate(dt['occ']))
        print '     There are {} samples in this batch'.format(L)
        shuffle = range(0,L)
        np.random.shuffle(shuffle)

        dt['ims'] = np.concatenate(dt['ims'])[shuffle]
        dt['geoms'] = np.concatenate(dt['geoms'])[shuffle]
        dt['occ'] = np.concatenate(dt['occ'])[shuffle]
        dt['int'] = np.concatenate(dt['int'])[shuffle]
        dt['subjects'] = np.concatenate(dt['subjects'])[shuffle]
        dt['tasks'] = np.concatenate(dt['tasks'])[shuffle]
        dt['poses'] = np.concatenate(dt['poses'])[shuffle]

        return dt

    def _filter_junk(self, ims, geoms, occ, int):
        # Check failed frontalization
        failed = np.asarray([i for i,x in enumerate(geoms) if np.sum(x)==0])

        # Filter out failed frontalization
        if len(failed)>0:
            print '     Filering {} junk samples'.format(len(failed))
            mask = np.ones(len(geoms), dtype=bool)
            mask[failed] = False
            ims, geom, occ, int = (ims[mask,...], geoms[mask,...], occ[mask,...], int[mask,...])

        return ims, geoms, occ, int

    def _read_au_intensities(self, root, subject, task):
        dt = [None] * len(self.aus_int)
        for au in self.aus_int:
            fname = self.path_int + root + subject + '_' + task + '_' + au + '_Int.csv'
            if os.path.exists(fname):
                dt.append([x[1] for x in read_csv(fname)])

        return np.asarray(dt)

    def _get_subjects(self):
        fnames = [f for f in os.listdir(self.path_ims) if f.endswith('.mp4')]
        return sorted(list(set([re.split('_', f)[2] for f in fnames])))

    def _get_tasks(self, subject):
        fnames = [f for f in os.listdir(self.path_ims) if subject in f and f.endswith('.mp4')]
        return sorted(list(set([re.split('_', f)[3] for f in fnames])))

def update_slices(slices, slice, idx):
    prefix  = list(slices[:slice])
    core = slices[slice]
    sufix = slices[slice+1:]

    core = list(np.reshape(core[:-1], (1, len(core[:-1]))))
    sufix = [x-1 for x in sufix]

    return prefix + core + sufix

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
    path = '/Users/cipriancorneanu/Research/data/fera2017'
    reader = ReaderFera2017(path)

    reader.read_geom(path, 'geom.pkl' )