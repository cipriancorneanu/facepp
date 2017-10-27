__author__ = 'cipriancorneanu'

import re
from facepp.processor.aligner import align
import cPickle
import scipy.io as io
from scipy.misc import imresize
from facepp.frontalizer.check_resources import check_dlib_landmark_weights
import dlib
from facepp.frontalizer.frontalize import ThreeD_Model
from facepp.processor.encoder import encode_parametric
from facepp.processor import partitioner
from extractor import extract, extract_face, _extend_rect, square_bbox
from reader import *
import time
from joblib import Parallel, delayed
import random
import h5py
import os.path
import matplotlib.pyplot as plt
from imgaug.imgaug import augmenters as iaa
import imgaug.imgaug as ia

class ReaderOuluCasia():
    def __init__(self, path):
        self.path = path
        self.illuminations = ['Strong', 'Weak', 'Dark']
        self.emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

    def read(self, opath, mpath, cores=2):
        # Load models
        check_dlib_landmark_weights(mpath + 'shape_predictor_models')
        predictor = dlib.shape_predictor(mpath + 'shape_predictor_models/shape_predictor_68_face_landmarks.dat')
        model3D = ThreeD_Model(mpath + 'frontalization_models/model3Ddlib.mat', 'model_dlib')
        eyemask = np.asarray(io.loadmat(mpath + 'frontalization_models/eyemask.mat')['eyemask'])

        # Get list of subjects
        for illumination in self.illuminations:
            print 'Illumination: {}'.format(illumination)
            subjects = os.listdir(self.path + illumination)
            print 'List of selected subjects: {}'.format(subjects)
            for subject in subjects:
                start_time = time.time()
                print 'Processing subject {}'.format(subject)

                dt = {'faces': [], 'geoms': [], 'emos': []}
                for emo in self.emotions:
                    target_path = self.path + illumination + '/' + subject + '/' + emo + '/'

                    # Read ims from path
                    if os.path.exists(target_path):

                        ims = read_folder(target_path)

                        print '     Extract faces and resize '
                        faces = Parallel(n_jobs=cores)(delayed(extract_face)(i,im) for i,im in enumerate(ims))

                        print '     Align faces'
                        aligned = [align(i, face, model3D, eyemask, predictor) if face_detected
                                   else (np.zeros((face.shape[0], face.shape[1], 3)), np.zeros((68,2)))
                                   for i,(face_detected,face) in enumerate(faces)]

                        afaces, ageoms = (np.asarray([x[0] for x in aligned], dtype=np.uint8),
                                                np.asarray([x[1] for x in aligned], dtype=np.float16))
                        emos = self._code_emo(emo)*np.ones(len(afaces),dtype=np.uint8)

                        # Filter out failed frontalization
                        afaces, ageoms, emos = self._filter_junk(afaces, ageoms, emos)

                        # Save
                        dt['faces'].append(afaces)
                        dt['geoms'].append(ageoms)
                        dt['emos'].append(emos)

                dt['faces'], dt['geoms'], dt['emos'] = (np.concatenate(dt['faces']), np.concatenate(dt['geoms']),
                                                            np.concatenate(dt['emos']))

                print '     Save data'
                cPickle.dump(dt, open(opath + illumination + '/' +  subject + '.pkl', 'wb'),
                             cPickle.HIGHEST_PROTOCOL)

                print '     Subject {} processsed in {:.2f} s'.format(subject, time.time() - start_time)

    def _filter_junk(self, faces, geoms, emos):
        # Check failed frontalization
        failed = np.asarray([i for i,x in enumerate(geoms) if np.sum(x)==0])

        # Filter out failed frontalization
        if len(failed)>0:
            print '     Filering {} junk samples'.format(len(failed))
            mask = np.ones(len(geoms), dtype=bool)
            mask[failed] = False
            faces, geoms, emos = (faces[mask,...], geoms[mask,...], emos[mask,...])

        return faces, geoms, emos

    def _code_emo(self, emo):
        map = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Sadness':4, 'Surprise':5}
        return int(map[emo])

class ReaderFera2017():
    def __init__(self, path):
        self.path = path
        self.path_ims = path + 'ims/'
        self.path_occ = path + 'occ/'
        self.path_int = path + 'int/'
        self.poses = [str(i) for i in range(1,10)]
        self.aus = [1, 4, 6, 7, 10, 12, 14, 15, 17, 23]
        self.aus_int = ['AU01', 'AU04', 'AU06', 'AU10', 'AU12', 'AU14', 'AU17']

        self.subjects = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                         'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020',
                         'F021', 'F022', 'F023', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007',
                         'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']

        self.folds3 = [['M016', 'F015', 'M005', 'F010', 'M009', 'F016', 'F001', 'M008', 'M013', 'M015', 'F017', 'F014', 'M010', 'F019'],
                        ['M011', 'F022', 'M007', 'M017', 'F003', 'M003', 'F004', 'M018', 'M006', 'F012', 'M002', 'M014', 'F018', 'F006'],
                        ['M004', 'F002', 'F009', 'F020', 'F007', 'F023', 'M012', 'M001', 'F021', 'F011', 'F013', 'F005', 'F008']]

        self.folds10 = [['M016', 'F015', 'M005', 'F010'],
                        ['M009', 'F016', 'F001', 'M008'],
                        ['M013', 'M015', 'F017', 'F014'],
                        ['M010', 'F019', 'M011', 'F022'],
                        ['M007', 'M017', 'F003', 'M003'],
                        ['F004', 'M018', 'M006', 'F012'],
                        ['M002', 'M014', 'F018', 'F006'],
                        ['M004', 'F002', 'F009', 'F020'],
                        ['F007', 'F023', 'M012', 'M001'],
                        ['F021', 'F011', 'F013', 'F005', ['F008']]]

    def read(self, opath, mpath, partition = 'train', poses=[6], cores=4, start=0, stop=1):
        if partition == 'train':
            root = 'FERA17_TR_'
        elif partition == 'validation':
            root = 'FERA17_VA_'

        subjects = self._get_subjects()
        subjects = [subjects[i] for i in range(start, stop)]
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
                        ims = read_video(vidname, colorspace='RGB')
                        print 'Reading video file {} in {}s'.format(vidname, (time.time() - start_time))

                        occ = read_csv(occname)
                        int = self._read_au_intensities(root, subject, task)

                        print '     Extract faces and resize '
                        faces = Parallel(n_jobs=cores)(delayed(extract_face)(i,im,ext=1.5,sz=224,verbose=True) for i,im in enumerate(ims))

                        print '     Align faces'
                        aligned = [align(i, face, model3D, eyemask, predictor, do_frontalize=False, verbose=True) if face_detected
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

                seq['int'][0] = [x for x in seq['int'][0][7:]]

                if self._valid_intensity(seq) and self._valid_occurence(seq):
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
                else:
                    print('     File {} does not have consistent valid labels. Ignore.'.format(f))

            if len(dt['occ'])>0:
                # Vectorize and shuffle
                dt = self._shuffle_data(dt)

                # Dump
                print 'Dumping batch {}'.format(bat)
                cPickle.dump(dt, open(self.path+'/fera17_' + str(bat), 'wb'), cPickle.HIGHEST_PROTOCOL)
            else:
                print 'This batch is empty'

        return dt

    def prepare(self, partition, pose, out_fname):
        path = self.path+partition+'/aligned/'+pose+'/'
        files_fera17 = sorted([f for f in os.listdir(path)])

        # Get data
        for subject_idx, subject in enumerate(self.subjects):
            # Load train data from aligned
            ims, lms, aus, ints, subjects, tasks, poses = ([], [], [], [], [], [], [])

            # Get files corresponding to subject
            subject_files = [f for f in files_fera17 if subject in f]
            print subject_files

            for fname in subject_files:
                print 'Loading file {}'.format(fname)
                dt = cPickle.load(open(path+fname, 'rb'))

                ims.append(dt['ims'])
                lms.append(dt['geoms'])

                # Append labels from BP4D
                task = fname.split('_')[2]
                aus_bp4d = np.asarray(read_csv(self.path+'/AU_OCC/'+subject+'_'+task+'.csv'))[1:,[1,2,4,6,7,10,12,14,15,17,23,24]]
                
                aus.append(aus_bp4d)
                ints.append(dt['int'])
                subjects.append(dt['subjects']*np.concatenate(dt['ims']).shape[0])
                poses.append(dt['poses']*np.concatenate(dt['ims']).shape[0])
                tasks.append(dt['tasks']*np.concatenate(dt['ims']).shape[0])

                
            (ims, lms, aus, ints, subjects, poses, tasks) = (np.squeeze(np.concatenate(ims, axis=1)), np.squeeze(np.concatenate(lms, axis=1)), \
                                               np.squeeze(np.concatenate(aus, axis=0)), np.squeeze(np.concatenate(ints, axis=1)),
                                               np.concatenate(subjects), np.concatenate(poses), np.concatenate(tasks))
            
            print 'Total number of samples is {}'.format(ims.shape)

            # Shuffle and split
            idx = np.random.permutation(ims.shape[0])
            slice_length = 256
            indices = [slice_length*x for x in range(1,int(np.ceil(ims.shape[0]/slice_length+1)))]
            splits = np.array_split(idx, indices)

            # Dump data
            if os.path.isfile(self.path+out_fname):
                file = h5py.File(self.path+out_fname, 'r+')
            else:
                file = h5py.File(self.path+out_fname, 'w')
                
            for i,x in enumerate(splits):
                segment = file.create_group(partition+'/'+pose+'/'+'subject_'+subject+'/segment_'+str(i))
                segment.create_dataset('faces', data=ims[x])
                segment.create_dataset('lms', data=lms[x])
                segment.create_dataset('aus', data=aus[x])
                segment.create_dataset('subjects', data=subjects[x])
                segment.create_dataset('poses', data=poses[x])
                segment.create_dataset('tasks', data=tasks[x])

    def prepare_patched_faces(self, partition, pose, out_fname, verbose=False):
        print 'Prepare patched faces for database {}'.format(out_fname)
        with h5py.File(self.path+out_fname, 'r+') as hf:
            for subject_k,subject_v in hf[partition+'/'+pose+'/'].items():
                print '{} of dataset {}'.format(segment_k, partition+'/'+pose+'/')

                for segment_k,segment_v in subject_v.items():

                    faces, lms = segment_v['faces'], segment_v['lms']

                    patches = []
                    for i, (face, lm) in enumerate(zip(faces, lms)):
                        patch = extract_patches(face, lm)
                        patches.append(patch)

                        '''
                        if i%100==0: print i
                        fig,ax = plt.subplots(1)
                        ax.imshow(face)
                        ax.scatter(lm[:,0], lm[:,1], color='g')
                        plt.show()
                        plt.imshow(patch)
                        plt.show()
                        '''
                        segment_v.create_dataset('faces_patched', data=np.concatenate(patches))

    def prepare_patches(self, partition, pose, out_fname, update=False):
        markers = {
            'leye': np.concatenate((np.arange(17,22), np.arange(36,42))),
            'reye': np.concatenate((np.arange(42,48), np.arange(22,27))),
            'nose': np.arange(27,36),
            'mouth': np.arange(48,68),
            'beye': np.asarray([21,22,42,28,39]),
            'lmouth': np.asarray([36,39,31,48]),
            'rmouth': np.asarray([42,45,35,54])
        }

        print 'Prepare patches for database {}'.format(out_fname)
        with h5py.File(self.path+out_fname, 'r+') as hf:
            for subject_k,subject_v in hf[partition+'/'+pose+'/'].items():
                for segment_k,segment_v in subject_v.items():
                    print '{} of dataset {}'.format(segment_k, partition+'/'+pose+'/'+subject_k)
                    faces, lms = segment_v['faces'], segment_v['lms']
                                
                    patches = {'leye':[], 'reye':[], 'beye':[], 'nose':[], 'mouth':[], 'lmouth':[], 'rmouth':[]}
                    for i, (face, lm) in enumerate(zip(faces, lms)):
                        # Extract patches
                        for k,v in markers.items():
                            if np.sum(lm)==0:
                                patch = np.zeros((56, 56, 3))
                            else:
                                patch = extract(face, square_bbox(lm[v]), extension=1.3, size=56)[0]
                                patches[k].append(patch)

                    for k,v in markers.items():
                        '''print partition+'/'+pose+'/'+subject_k+'/'+segment_k+'/'+k'''
                        
                        if partition+'/'+pose+'/'+subject_k+'/'+segment_k+'/'+k in hf:
                            data = segment_v[k]
                            data[...] = np.asarray(patches[k])
                        else:
                            segment_v.create_dataset(k, data=np.asarray(patches[k]))

    def augment(self, n_augm, out_fname, update=False):
        # Define augmenter
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-20, 20), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),

                sometimes(
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5) # add gaussian noise to images
                ),
            ],
            random_order=True
        )
                
        with h5py.File(self.path+out_fname, 'r+') as hf:
            for subject_k,subject_v in hf['train/pose6/'].items():
                if subject_k.split('_')[1] in ['F017', 'F018', 'F019','F020']:
                    for segment_k,segment_v in subject_v.items():
                        labels = np.asarray(segment_v['aus'])
                        idxs = self.balance(labels)
                        labels = labels[idxs]

                        for i in range(n_augm):
                            node  = 'train/pose6/'+subject_k+'/'+segment_k+'_'+str(i)
                            _ = hf.create_group(node)

                        for tp in ['faces', 'leye', 'reye', 'beye', 'nose', 'mouth', 'lmouth', 'rmouth']:
                            print '{}/{}/{}'.format(subject_k, segment_k, tp)
                            images = np.asarray(segment_v[tp])
                            if np.max(idxs)<images.shape[0]:
                                images = images[idxs]
                            else:
                                print ('-------An index error-------')
                                labels = np.asarray(segment_v['aus'])
                                
                            for i in range(n_augm):
                                augm_images = seq.augment_images(images)
                                node  = 'train/pose6/'+subject_k+'/'+segment_k+'_'+str(i)
                                print '     Writing node {} with data of shape {}'.format(node, augm_images.shape)
                                segment = hf[node]
                                
                                segment.create_dataset(tp, data=augm_images)
                                if tp=='faces':
                                    segment.create_dataset('aus', data=labels)
                                
    def balance(self, batch):
        b_sz, n = batch.shape
        p = [0.21, 0.17, 0.20, 0.46, 0.55, 0.59, 0.56, 0.46, 0.17, 0.34, 0.16, 0.14]

        pmark = np.asarray([ x/y if y>0 else 1 for x,y in zip(np.sum(batch * np.tile(p, (b_sz, 1)), axis=1),
                                                             [len(np.where(x==1)[0]) for x in batch])])
        sort_pmark = np.argsort(pmark)

        idxs = np.tile(sort_pmark[:b_sz/2], 2)
        return idxs

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

        print [x.shape for x in dt['int']]
        
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

    def _valid_intensity(self, seq):
        seq_length = len(np.asarray(seq['occ'][0]))

        if len(np.asarray(seq['int'][0]).shape)<2:
            print ('\tNo intensity labels.')
            return False
        elif np.asarray(seq['int'][0]).shape[1]!=seq_length:
            print('\tIntensity labels number doesn\'t match the other labels')
            return False
        elif np.asarray(seq['int'][0]).shape[0]!=7:
            print('\tNumber of intensity classes is lower than expected.')
            return False
        else:
            return True

    def _valid_occurence(self, seq):
        occ = np.asarray(seq['occ'][0], dtype=np.uint8)
        max_value = np.amax(occ)
        
        if max_value>1:
            print('\tMax value of occurence is {}'.format(max_value))
            return False
        else:
            return True

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
        #Pass sample from text format to vectorial format
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
        self.path_aligned_left = path + 'aligned_left/'
        self.path_aligned_right = path + 'aligned_right/'
        self.path_lm = path + 'Landmark_Points/'
        self.path_au = path + 'ActionUnit_Labels/'
        self.path_vidl = path + 'Video_LeftCamera/'
        self.path_vidr = path + 'Video_RightCamera/'

    def read(self, fname, mpath, start, stop, video='left', cores=4):
        if os.path.exists(self.path+fname):
            return cPickle.load(open(self.path+fname, 'rb'))

        subjects = sorted([f for f in os.listdir(self.path_au)])
        subjects = [subjects[i] for i in range(start, stop)]
        print 'List of subjects: {}'.format(subjects)

        # Load alignment models
        check_dlib_landmark_weights(mpath + 'shape_predictor_models')
        predictor = dlib.shape_predictor(mpath + 'shape_predictor_models/shape_predictor_68_face_landmarks.dat')
        model3D = ThreeD_Model(mpath + 'frontalization_models/model3Ddlib.mat', 'model_dlib')
        eyemask = np.asarray(io.loadmat(mpath + 'frontalization_models/eyemask.mat')['eyemask'])

        print('###### READING DISFA DATASET ######')
        # TODO: Check SN004, SN005 that lack on au and the 5 that fail (7 in total)
        for subject in subjects:
            print('Reading subject ' + subject)

            dt = {'images':[], 'landmarks':[], 'aus':[], 'subjects':[]}
            lm_seq = np.asarray([np.fliplr(x['pts'])
                                 for x in read_folder(self.path_lm+subject+'/'+'frame_lm/', self._lm_file_sorter)])
            au_seq = read_folder(self.path_au+subject+'/', self._au_file_sorter)

            if video=='left':
                im_seq = read_video(self.path_vidl+'LeftVideo'+subject+'_comp.avi')
            elif video=='right':
                im_seq = read_video(self.path_vidr+'RightVideo'+subject+'_comp.avi')

            # Extract face and resize
            print '     Extract faces and resize '
            faces = Parallel(n_jobs=cores)(delayed(extract_face)(i,im,ext=1.5,sz=224,verbose=True) for i,im in enumerate(im_seq))

            print '     Align faces'
            aligned = [align(i, face, model3D, eyemask, predictor, do_frontalize=False, verbose=True) if face_detected
                       else (np.zeros((face.shape[0], face.shape[1], 3)), np.zeros((68,2)))
                       for i,(face_detected,face) in enumerate(faces)]

            afaces, ageoms = (np.asarray([x[0] for x in aligned], dtype=np.uint8),
                              np.asarray([x[1] for x in aligned], dtype=np.float16))

            dt['images'].append(afaces)
            dt['landmarks'].append(ageoms)

            dt['aus'].append(self._vectorize_au_sequence(au_seq))
            dt['subjects'].append(subject)

            print('Dumping subject {}'.format(subject))
            if video=='left':
                file = h5py.File(self.path+'/aligned_left/disfa_subject_' + str(subject)+'.h5', 'w')
            elif video=='right':
                file = h5py.File(self.path+'/aligned_right/disfa_subject_' + str(subject)+'.h5', 'w')

            grp = file.create_group('dt')
            for k,v in dt.items():
                grp.create_dataset(k, data=v)

        return dt

    def get_subject(self, fname):
        return fname.split('_')[2].split('.')[0]

    def prepare(self, partition, pose, out_fname):
        if pose == 'pose0': in_path = self.path_aligned_left
        elif pose == 'pose1': in_path = self.path_aligned_right

        # Get files
        test_files = ['disfa_subject_SN030.h5', 'disfa_subject_SN031.h5', 'disfa_subject_SN032.h5']
        if partition == 'train':
            files = list(set(sorted([f for f in os.listdir(in_path)])) - set(test_files))
        elif partition == 'test':
            files = test_files
        print 'List of files: {}'.format(files)

        # Load data
        ims, lms, aus, subjects, poses = ([], [], [], [], [])
        for fname in files:
            print 'Loading file {}'.format(fname)
            with h5py.File(in_path + fname, 'r') as hf:
                ims.append(hf['dt']['images'][()])
                lms.append(hf['dt']['landmarks'][()])
                aus.append(hf['dt']['aus'][()])
                subjects.append([self.get_subject(fname)]*hf['dt']['images'][()].shape[1])
                poses.append(['left']*hf['dt']['images'][()].shape[1])

        (ims, lms, aus, subjects, poses) = (np.squeeze(np.concatenate(ims, axis=1)), np.squeeze(np.concatenate(lms, axis=1)), \
                                           np.squeeze(np.concatenate(aus, axis=1)), np.concatenate(subjects),
                                           np.concatenate(poses))

        print 'Total number of samples is {}'.format(ims.shape)

        # Shuffle and split
        idx = np.random.permutation(ims.shape[0])
        slice_length = 1024
        indices = [slice_length*x for x in range(1,int(np.ceil(ims.shape[0]/slice_length+1)))]
        splits = np.array_split(idx, indices)

        # Dump data
        if os.path.isfile(self.path+out_fname):
            file = h5py.File(self.path+out_fname, 'r+')
        else:
            file = h5py.File(self.path+out_fname, 'w')

        grp = file.create_group(partition+'/'+pose+'/')
        for i,x in enumerate(splits):
            segment = grp.create_group('segment_'+str(i))
            segment.create_dataset('faces', data=ims[x])
            segment.create_dataset('lms', data=lms[x])
            #segment.create_dataset('aus', data=aus[x]) # 2 AUS are missing from SN004 and SN005
            segment.create_dataset('subjects', data=subjects[x])
            segment.create_dataset('poses', data=poses[x])

    def prepare_patched_face(self, partition, pose, out_fname):
        # From face creates masked version containing fiducial patches
        print 'Prepare patched faces for database {}'.format(out_fname)
        with h5py.File(self.path+out_fname, 'r+') as hf:
            for segment_k,segment_v in hf[partition+'/'+pose+'/'].items():
                print '{} of dataset {}'.format(segment_k, partition+'/'+pose+'/')
                faces, lms = segment_v['faces'], segment_v['lms']

                patches = []
                for i, (face, lm) in enumerate(zip(faces, lms)):
                    patch = extract_patches(face, lm)
                    patches.append(patch)

                print np.asarray(patches).shape
                segment_v.create_dataset('faces_patched', data=np.concatenate(patches))

                '''
                if i%100==0: print i
                fig,ax = plt.subplots(1)
                ax.imshow(face)
                ax.scatter(lm[:,0], lm[:,1], color='g')
                plt.show()
                plt.imshow(patches)
                plt.show()
                '''

    def prepare_patches(self, partition, pose, out_fname):
        markers = {
            'leye': np.concatenate((np.arange(17,22), np.arange(36,42))),
            'reye': np.concatenate((np.arange(42,48), np.arange(22,27))),
            'nose': np.arange(27,36),
            'mouth': np.arange(48,68),
            }

        print 'Prepare patches for database {}'.format(out_fname)
        with h5py.File(self.path+out_fname, 'r+') as hf:
            for segment_k,segment_v in hf[partition+'/'+pose+'/'].items():
                if 'leye' not in segment_v:
                    print '{} of dataset {}'.format(segment_k, partition+'/'+pose+'/')
                    faces, lms = segment_v['faces'], segment_v['lms']

                    patches = {'leye':[], 'reye':[], 'nose':[], 'mouth':[]}
                    for i, (face, lm) in enumerate(zip(faces, lms)):
                        # Extract patches
                        for k,v in markers.items():
                            if np.sum(lm)==0:
                                patch = np.zeros((56, 56, 3))
                            else:
                                patch = extract(face, square_bbox(lm[v]), extension=1.3, size=56)[0]
                            patches[k].append(patch)

                    for k,v in markers.items():
                        segment_v.create_dataset(k, data=np.asarray(patches[k]))

    def _vectorize_au_sequence(self, au_seq):
        return np.asarray(np.transpose(np.vstack([ [int(x[0].split(',')[1]) for x in au] for au in au_seq])),
                          dtype = np.uint8)

    def _au_file_sorter(self, files):
        # Sort files according to integer key in file name
        sorting = list(np.argsort([int(f.split('au')[1].split('.')[0]) for f in files]))
        return  [files[s] for s in sorting]

    def _lm_file_sorter(self, files):
        return sorted(files)

def generate_ml_mnist(visualize=False):
    # Random shifts
    import keras
    from keras.datasets import mnist
    from keras.preprocessing.image import ImageDataGenerator
    from matplotlib import pyplot
    from keras import backend as K

    K.set_image_dim_ordering('th')

    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape to be [samples][pixels][width][height]
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    # convert from int to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # define data preparation
    datagen = ImageDataGenerator(rotation_range=30)

    # fit parameters from data
    datagen.fit(x_train)

    x_train_ml = np.zeros((len(x_train), 64, 64), dtype = np.uint8)
    x_test_ml = np.zeros((len(x_test), 64, 64), dtype = np.uint8)
    y_train_ml, y_test_ml = ([], [])

    # configure batch size and retrieve one batch of images
    visualize = False
    for x,y,x_ml,y_ml in [(x_train, y_train, x_train_ml, y_train_ml),
                          (x_test, y_test, x_test_ml, y_test_ml)]:
        # number of samples to generate
        for i, (x_batch, y_batch) in enumerate(datagen.flow(x, y, batch_size=30)):
            if i % 1000 == 0: print 'Generate sample {}'.format(i)

            y_batch, idxs = np.unique(y_batch, return_index=True)

            im, targets = batch_patcher(np.squeeze(x_batch[idxs]), y_batch)

            x_ml[i,...] = im
            y_ml.append(targets)

            # show the plot
            if visualize:
                print 'Targets: {}'.format(targets)
                pyplot.imshow(im, cmap=pyplot.get_cmap('gray'))
                pyplot.savefig('/Users/cipriancorneanu/Research/code/afea/results/ml_mnist_test' + str(i) +'.png')

            if i == 10000-1: break
            pass

    # Pass targets to categorical
    y_train_ml = np.asarray([np.sum(keras.utils.to_categorical(x, 10), axis=0) for x in y_train_ml], dtype=np.uint8)
    y_test_ml = np.asarray([np.sum(keras.utils.to_categorical(x, 10), axis=0) for x in y_test_ml], dtype=np.uint8)

    data = [(np.expand_dims(x_train_ml, axis=3), y_train_ml),
            (np.expand_dims(x_test_ml, axis=3), y_test_ml)]

    path = '/Users/cipriancorneanu/Research/data/'
    cPickle.dump(data, open(path + 'ml_mnist_test.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

def batch_patcher(x, y, shape=(64,64)):
    '''
    Patch batch of images inside shape with variations (random pooling, scaling, positions)
    '''
    # Generate positions
    positions = generate_grid_positions(shape)

    # Pick n samples out of batch
    idxs = sample(low=0, high=len(x))
    x, y, positions = x[idxs], y[idxs], positions[y][idxs]

    # Randomly rescale between high and low
    low, high= (0.5, 1.2)
    patches = [imresize(x, (high-low)*np.random.random()+low) for x in x]

    # Patch
    return patch(patches, positions, shape), y

def sample(low=0, high=10):
    '''
    Generate random number of random integers from the normal distribution in the interval [low, high]
    '''
    miu, sigma = (high-low)/2, 0.1*(high-low)
    n = max(low, min(int(sigma * np.random.randn() + miu), high))

    return random.sample(xrange(low,high), n)

def generate_grid_positions(region):
    '''
    Generate set of 10 positions in predefined rectangular region
    '''

    # First two rows will contain 3 equally spaces positions each
    step_x, step_y = (region[0]/3, region[1]/3)
    offset_x, offset_y = (region[1]/6, region[0]/6)
    grid = [ (offset_y + step_y*(i//3), offset_x + step_x*(i%3)) for i in range(0,6)]

    # Recompute for last row. It will contain remaining 4 positions
    step_x, step_y = (region[0]/4, region[1]/3)
    offset_x, offset_y = (region[1]/8, region[0]/6)
    grid =  grid + [ (offset_y + step_y*i, offset_x + step_x*j) for i,j in [(2,0), (2,1), (2,2), (2,3)]]

    # Add noise to grid
    noise = np.zeros_like(grid)

    return grid + noise

def generate_random_positions(roi):
    pass

def patch(patches, positions, shape=(64,64)):
    '''
    Patch patches at positions inside shape
    :param patches: Set of image patches
    :param positions: Positions to patch
    :param shape: Shape of the output
    :return:
    '''
    output = np.zeros((len(patches), shape[0], shape[1]))
    center = np.divide(shape,2)

    for i, (out, patch, pos) in enumerate(zip(output, patches, positions)):
        # Compute ROI
        roi = np.concatenate([pos - [x//2 for x in patch.shape], pos - [x//2 for x in patch.shape] + patch.shape])

        corners = [p for p in [[roi[0], roi[1]], [roi[0], roi[2]], [roi[2], roi[1]], [roi[2], roi[3]]]]

        extreme = np.argmax([np.sqrt(np.sum((c-center)**2)) for c in corners])

        sz = np.max(np.abs(corners[extreme]-center))*2

        if sz > shape[0]:
            out_ext = np.zeros((sz, sz))
        else:
            out_ext = out

        # If on the negative side translate
        if len(np.where(roi<0)[0])>0:
            roi = roi - [np.min([np.min(np.take(roi, [0,2])), 0]), np.min([np.min(np.take(roi, [1,3])), 0]),
                         np.min([np.min(np.take(roi, [0,2])), 0]), np.min([np.min(np.take(roi, [1,3])), 0])]

        # Patch in image
        out_ext[roi[0]:roi[2],roi[1]:roi[3]] = patch

        if sz > shape[0]:
            output[i,...] = out_ext[sz/2-center[0]:sz/2+center[0], sz/2-center[1]:sz/2+center[1]]
        else:
            output[i,...] = out_ext

    return np.asarray(np.clip(np.sum(output, axis=0), 0, 255), dtype=np.uint8)


def extract_patches(face, geom):
        markers = {
            'leye': np.concatenate((np.arange(17,22), np.arange(36,42))),
            'reye': np.concatenate((np.arange(42,48), np.arange(22,27))),
            'nose': np.arange(27,36),
            'mouth': np.arange(48,68),
        }

        patches = np.zeros_like(face)
        for (_,v) in markers.items():
            bbox = _extend_rect([int(min(geom[v, 0])), int(min(geom[v, 1])),
                                 int(max(geom[v, 0])), int(max(geom[v, 1]))], 1.5)

            # Copy content
            mask = np.zeros_like(face, dtype=np.uint8)
            for i in range(bbox[0], bbox[2]):
                for j in range(bbox[1], bbox[3]):
                        patches[j,i] = face[j,i]

        return patches

if __name__ == '__main__':
    reader = ReaderFera2017('/data/data1/datasets/fera2017/')
    reader.augment(n_augm=4, out_fname='bp4d_augm.h5')
