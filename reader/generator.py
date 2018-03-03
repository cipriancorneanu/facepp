__author__ = 'cipriancorneanu'

import cPickle
import h5py
import numpy as np
import gc
from sklearn.utils import shuffle
from imgaug.imgaug import augmenters as iaa
import scipy
from  scipy.misc import imresize

class GeneratorPreds():
    def __init__(self, path):
        self.path = path

    def generate(self, partition, type, batch_size=32):
        with h5py.File(self.path+'/fera17_preds.h5', 'r') as hf:
            data = [hf[partition+'/'+t] for t in type]
            for i in range(0, data[0]['gt'].shape[0]/batch_size):
                dvdv = []
                for dt in data:
                    dvdv.append((dt['gt'][(i)*batch_size:(i+1)*batch_size],
                                    dt['pred'][(i)*batch_size:(i+1)*batch_size]))
                yield tuple(dvdv)

class GeneratorPredsLC():
    def __init__(self, path):
        self.path = path

    def generate(self, partition, type, batch_size=32):
        with h5py.File(self.path+'/label_coding_preds.h5', 'r') as hf:
            data = [hf[partition+'/'+t] for t in type]
            for i in range(0, data[0]['gt'].shape[0]/batch_size):
                dvdv = []
                for dt in data:
                    dvdv.append((dt['gt'][(i)*batch_size:(i+1)*batch_size],
                                 dt['pred'][(i)*batch_size:(i+1)*batch_size]))
                yield tuple(dvdv)
                
class Generator():
    def __init__(self, path, db, type):
        self.path = path
        self.db = db
        self.type = type

    def _get_segments(self, partition, pose, patch_wise=True):
        print partition
        if partition == 'train':
            if self.db == 'disfa':
                if pose=='all':
                    databases = [{'fname':'disfa/disfa.h5', 'datasets':[partition+'/pose0', partition+'/pose1']}]
                else:
                    databases = [{'fname':'disfa/disfa.h5', 'datasets':[partition+'/'+pose]}]
            elif self.db == 'fera':
                if pose=='all':
                    databases = [{'fname':'fera/fera17.h5', 'datasets':[partition+'/pose5', partition+'/pose6']}]
                else:
                    databases = [{'fname':'fera/fera17.h5', 'datasets':[partition+'/'+pose]}]
            elif self.db == 'all':
                databases = [
                    {'fname':'disfa/disfa.h5', 'datasets':[partition+'/pose0', partition+'/pose1']},
                    {'fname':'fera/fera17.h5', 'datasets':[partition+'/pose5', partition+'/pose6']}
                ]
        elif partition == 'validation':
            if self.db == 'disfa':
                if pose=='all':
                    databases = [{'fname':'disfa/disfa.h5', 'datasets':['test/pose0', 'test/pose1']}]
                else:
                    databases = [{'fname':'disfa/disfa.h5', 'datasets':['test/'+pose]}]
            elif self.db == 'fera':
                if pose=='all':
                    databases = [{'fname':'fera/fera17.h5', 'datasets':['test/pose5', 'test/pose6']}]
                else:
                    databases = [{'fname':'fera/fera17.h5', 'datasets':['test/'+pose]}]
            elif self.db == 'all':
                databases = [
                    {'fname':'disfa/disfa.h5', 'datasets':['test/pose0', 'test/pose1']},
                    {'fname':'fera/fera17.h5', 'datasets':['test/pose5', 'test/pose6']}
                ]
        elif partition == 'test':
            databases = [{'fname':'external_test.h5', 'datasets':['/data']}]

        if self.type == 'all':
            data_types = ['faces', 'faces_patched', 'leye', 'reye', 'mouth', 'nose']
        else:
            data_types = [self.type]

        segments = []
        for db in databases:
            print db 
            print db['fname']
            with h5py.File(self.path+db['fname'], 'r') as hf:
                for ds in db['datasets']:
                    for k,v in hf[ds].items():
                        if patch_wise:
                            np.random.shuffle(data_types)
                            for dt_type in data_types:
                                s = {'db':db['fname'], 'ds':ds, 'segm': k, 'type':dt_type}
                                segments.append(s)
                        else:
                            s = {'db':db['fname'], 'ds':ds, 'segm': k}
                            segments.append(s)
        return segments
                                
    def _get_batches_(self, segment, mini_batch_size, with_labels):
        batches = []
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['segm']]
            if segment['type']=='faces_patched':
                faces_patched = np.reshape(v['faces_patched'], (-1, 224, 224, 3))
                for i in range(0, faces_patched.shape[0]/mini_batch_size):
                    if with_labels:
                        batches.append((faces_patched[(i)*mini_batch_size:(i+1)*mini_batch_size],
                                        v['aus'][(i)*mini_batch_size:(i+1)*mini_batch_size]))
                    else:
                        batches.append(faces_patched[(i)*mini_batch_size:(i+1)*mini_batch_size])
            else:
                for i in range(0, v[segment['type']].shape[0]/mini_batch_size):
                    ims = np.asarray([imresize(im, (224,224)) for im in v[segment['type']][(i)*mini_batch_size:(i+1)*mini_batch_size]], dtype=np.uint8)
                    if with_labels:
                        batches.append((ims, v['aus'][(i)*mini_batch_size:(i+1)*mini_batch_size]))
                    else:
                        batches.append(ims)
        return batches

    #TODO: create get_batches function that returns selectable patches
    def _get_batches(self, segment, mini_batch_size, with_labels, patch_wise=True):
        data_types = ['faces', 'leye', 'beye', 'nose', 'mouth']
        
        batches = []
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['segm']]
            
            if patch_wise:
                for i in range(0, v[segment['type']].shape[0]/mini_batch_size):
                    batch = []
                    for tp in segment['type']:
                        if tp == 'faces_patched':
                            ims = np.reshape(v['faces_patched'], (-1, 224, 224, 3))
                        else:
                            ims = np.asarray([imresize(im, (224,224)) for im in v[tp][(i)*mini_batch_size:(i+1)*mini_batch_size]], dtype=np.uint8)
                        batch.append(ims)

                    if with_labels:
                        batch.append(v['aus'][(i)*mini_batch_size:(i+1)*mini_batch_size])
                    batches.append(tuple(batch))
            else:
                for i in range(0, v['faces'].shape[0]/mini_batch_size):
                    accumulate_batch, valid = [], True

                    for tp in data_types:
                        ims = np.asarray([imresize(im, (224,224)) for im in v[tp][(i)*mini_batch_size:(i+1)*mini_batch_size]], dtype=np.uint8)
                        
                        if ims.shape[0] < mini_batch_size:
                            valid = False

                        if not with_labels:
                            accumulate_batch.append(ims)
                        else:
                            accumulate_batch.append((v['aus'][(i)*mini_batch_size:(i+1)*mini_batch_size], ims))
                
                    if valid:
                        batches.append(tuple(accumulate_batch))
                    else:
                        print 'Ignore this batch. Patch dimensions do not coincide.'
                    
        return batches
                                
    def _get_batches_meta(self, segment, mini_batch_size):
        batches = []
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['segm']]
            for i in range(1, v[segment['type']].shape[0]/mini_batch_size):
                batches.append(
                    (v[segment['type']][(i-1)*mini_batch_size:i*mini_batch_size],
                     v['lms'][(i-1)*mini_batch_size:i*mini_batch_size],
                     v['aus'][(i-1)*mini_batch_size:i*mini_batch_size],
                     v['subjects'][(i-1)*mini_batch_size:i*mini_batch_size],
                     v['poses'][(i-1)*mini_batch_size:i*mini_batch_size],
                     v['tasks'][(i-1)*mini_batch_size:i*mini_batch_size]))
        return batches

    def generate(self, partition, pose='all', batch_size=32, with_labels=False, patch_wise=True, shuffle=True):
        segments = self._get_segments(partition, pose, patch_wise=patch_wise)

        if shuffle:
            np.random.shuffle(segments)

        while True:
            for s in segments:
                print s
                for b in self._get_batches(s, batch_size, with_labels=with_labels, patch_wise=patch_wise):
                    yield b
                    
    def n_samples_train(self, partition='all'):
        disfa = GeneratorDisfa(self.path + 'disfa/', 'disfa', self.type)
        fera = GeneratorFera(self.path + 'fera/', 'fera', self.type)

        if self.type == 'all':
            if self.db == 'fera':
                return 6*fera.n_samples_train(partition)
            if self.db == 'disfa':
                return 6*disfa.n_samples_train(partition)
            if self.db == 'all':
                return 6*(fera.n_samples_train(partition)+disfa.n_samples_train(partition))
        else:
            if self.db == 'fera':
                return  fera.n_samples_train(partition)
            if self.db == 'disfa':
                return  disfa.n_samples_train(partition)
            if self.db == 'all':
                return (fera.n_samples_train(partition)+disfa.n_samples_train(partition))
            
    def n_samples_validation(self, partition='all'):
        disfa = GeneratorDisfa(self.path + 'disfa/', 'disfa', self.type)
        fera = GeneratorFera(self.path + 'fera/', 'disfa', self.type)
        
        if self.type == 'all':
            if self.db == 'fera':
                return 6*fera.n_samples_validation(partition)
            if self.db == 'disfa':
                return 6*disfa.n_samples_validation(partition)
            if self.db == 'all':
                return 6*(fera.n_samples_validation(partition)+disfa.n_samples_validation(partition))
        else:
            if self.db == 'fera':
                return fera.n_samples_validation(partition)
            if self.db == 'disfa':
                return disfa.n_samples_validation(partition)
            if self.db == 'all':
                return disfa.n_samples_validation(partition)+fera.n_samples_validation(partition)
            
    def n_samples_test(self):
        return 32

class GeneratorBP4D():
    def __init__(self, path, type, n_folds):
        self.path = path
        self.type = type

        if type == 'all':
            self.n_patches = 8
        else:
            self.n_patches = len(type)
            
        if n_folds == 3:
            self.get_subjects = self._get_subject_list_3fold
        elif n_folds == 10:
            self.get_subjects = self._get_subject_list_10fold

    def generate(self, fold, batch_size=32, with_labels=False, augment=False, shuffle=True, patch_wise=False, verbose=True):
        segments = self._get_segments(self.get_subjects(fold), augment, patch_wise)
        if shuffle: np.random.shuffle(segments)

        while True:
            for s in segments:
                if verbose == True:
                    print s
                for b in self._get_batches(s, batch_size, patch_wise, with_labels):
                    yield b
                    
    def _get_subject_list_3fold(self, fold):
        if fold==1:
            return ['M016', 'F015', 'M005', 'F010', 'M009', 'F016', 'F001', 'M008', 'M013', 'M015', 'F017', 'F014', 'M010', 'F019']
        elif fold==2:
            return ['M011', 'F022', 'M007', 'M017', 'F003', 'M003', 'F004', 'M018', 'M006', 'F012', 'M002', 'M014', 'F018', 'F006']
        elif fold==3:
            return ['M004', 'F002', 'F009', 'F020', 'F007', 'F023', 'M012', 'M001', 'F021', 'F011', 'F013', 'F005', 'F008']

    def _get_subject_list_10fold(self, fold):
        if fold==1:
            return ['M016', 'F015', 'M005', 'F010']
        elif fold==2:
            return ['M009', 'F016', 'F001', 'M008']
        elif fold==3:
            return ['M013', 'M015', 'F017', 'F014']
        elif fold==4:
            return ['M010', 'F019','M011', 'F022']
        elif fold==5:
            return ['M007', 'M017', 'F003', 'M003']
        elif fold==6:
            return ['F004', 'M018', 'M006', 'F012']
        elif fold==7:
            return ['M002', 'M014', 'F018', 'F006']
        elif fold==8:
            return ['M004', 'F002', 'F009', 'F020']
        elif fold==9:
            return ['F007', 'F023', 'M012', 'M001']
        elif fold==10:
            return ['F021', 'F011', 'F013', 'F005', 'F008']


    def _get_segments(self, subject_list, augm, patch_wise=True):
        if augm:
            databases = [{'fname':'bp4d_augm.h5', 'datasets':['/train/pose6']}]
        else:
            databases = [{'fname':'bp4d.h5', 'datasets': ['/train/pose6']}]

        if self.type == 'all':
            data_types = ['faces', 'leye', 'reye', 'beye', 'mouth', 'nose', 'lmouth', 'rmouth']
        else:
            data_types = self.type

        segments = []
        for db in databases:
            with h5py.File(self.path+db['fname'], 'r') as hf:
                for ds in db['datasets']:
                    for subject in subject_list:
                        for segment_k, segment_v in hf[ds+'/subject_'+subject].items():
                            if patch_wise:
                                np.random.shuffle(data_types)
                                for dt_type in data_types:
                                    s = {'db':db['fname'], 'ds':ds, 'subject':'/subject_'+subject, 'segm':'/'+segment_k, 'type':dt_type}
                                    segments.append(s)
                            else:
                                s = {'db':db['fname'], 'ds':ds, 'subject':'/subject_'+subject, 'segm':'/'+segment_k}
                                segments.append(s)
        return segments

    def _get_batches(self, segment, mini_batch_size, patch_wise=True, with_labels=False):
        data_types = ['faces', 'leye', 'reye', 'beye', 'nose', 'mouth', 'lmouth', 'rmouth']

        batches = []
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['subject']+segment['segm']]

            if patch_wise:
                for i in range(0, v[segment['type']].shape[0]/mini_batch_size):
                    ims = np.asarray([imresize(im, (224,224)) for im in v[segment['type']][(i)*mini_batch_size:(i+1)*mini_batch_size]], dtype=np.uint8)
                    if not with_labels:
                        batches.append(ims)
                    else:
                        batches.append((ims, v['aus'][(i)*mini_batch_size:(i+1)*mini_batch_size]))
            else:
                for i in range(0, v['faces'].shape[0]/mini_batch_size):
                    accumulate_batch, valid = [], True
                    for type in data_types:
                        ims = np.asarray([imresize(im, (224,224)) for im in v[type][(i)*mini_batch_size:(i+1)*mini_batch_size]], dtype=np.uint8)

                        if ims.shape[0] < mini_batch_size:
                            valid = False
                        
                        if not with_labels:
                            accumulate_batch.append(ims)
                        else:
                            accumulate_batch.append((v['aus'][(i)*mini_batch_size:(i+1)*mini_batch_size], ims))
                    if valid:
                        batches.append(tuple(accumulate_batch))
                    else:
                        print 'Ignore this batch. Patch dimensions do not coincide.'
        return batches

    def n_samples_fold(self, fold, augm):
        n = 0
        with h5py.File(self.path+'bp4d.h5', 'r') as hf:
            for subject in self.get_subjects(fold):
                for k,v in hf['train/pose6/'+'subject_'+subject].items():
                    n += v['faces'].shape[0]
        if augm:
            return 4*n*self.n_patches
        else:
            return n*self.n_patches


class GeneratorDisfa():
    def __init__(self, path, type, n_folds):
        self.path = path
        self.type = type

        if type == 'all':
            self.n_patches = 5
        else:
            self.n_patches = len(type)
            
        if n_folds == 3:
            self.get_subjects = self._get_subject_list_3fold
        else:
            print 'This number of folds is not set yet.'
            
    def generate(self, fold, batch_size=32, with_labels=False, augment=False, shuffle=True, patch_wise=False, verbose=True):
        segments = self._get_segments(self.get_subjects(fold), augment, patch_wise)
        if shuffle: np.random.shuffle(segments)

        while True:
            for s in segments:
                if verbose == True:
                    print s
                for b in self._get_batches(s, batch_size, patch_wise, with_labels):
                    yield b

    def _get_subject_list_3fold(self, fold):
        if fold==1:
            return ['SN001', 'SN005', 'SN009', 'SN013', 'SN021', 'SN026', 'SN030', 'SN004', 'SN008']
        elif fold==2:
            return ['SN002', 'SN006', 'SN010', 'SN016', 'SN023', 'SN027', 'SN031', 'SN012', 'SN018']
        elif fold==3:
            return ['SN003', 'SN007', 'SN011', 'SN017', 'SN024','SN028', 'SN032','SN025', 'SN029']

    def _get_segments(self, subject_list, augm=0, patch_wise=True):
        if augm:
            print 'There is no augmentation available. I will load the normal data.'
            databases = [{'fname':'disfa.h5', 'datasets':['pose0']}]
        else:
            databases = [{'fname':'disfa.h5', 'datasets': ['pose0']}]

        if self.type == 'all':
            data_types = ['faces', 'leye', 'beye', 'mouth', 'nose']
        else:
            data_types = self.type

        segments = []
        for db in databases:
            with h5py.File(self.path+db['fname'], 'r') as hf:
                for ds in db['datasets']:
                    for subject in subject_list:
                        for segment_k, segment_v in hf[ds+'/subject_'+subject].items():
                            if patch_wise:
                                np.random.shuffle(data_types)
                                for dt_type in data_types:
                                    s = {'db':db['fname'], 'ds':ds, 'subject':'/subject_'+subject, 'segm':'/'+segment_k, 'type':dt_type}
                                    segments.append(s)
                            else:
                                s = {'db':db['fname'], 'ds':ds, 'subject':'/subject_'+subject, 'segm':'/'+segment_k}
                                segments.append(s)
        return segments


    def _get_batches(self, segment, mini_batch_size, patch_wise=True, with_labels=False):
        data_types = ['faces', 'leye', 'beye', 'nose', 'mouth']

        batches = []
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['subject']+segment['segm']]

            if patch_wise:
                for i in range(0, v[segment['type']].shape[0]/mini_batch_size):
                    ims = np.asarray([imresize(im, (224,224)) for im in v[segment['type']][(i)*mini_batch_size:(i+1)*mini_batch_size]], dtype=np.uint8)
                    if not with_labels:
                        batches.append(ims)
                    else:
                        batches.append((ims, v['aus'][(i)*mini_batch_size:(i+1)*mini_batch_size]))
            else:
                for i in range(0, v['faces'].shape[0]/mini_batch_size):
                    accumulate_batch, valid = [], True
                    for type in data_types:
                        ims = np.asarray([imresize(im, (224,224)) for im in v[type][(i)*mini_batch_size:(i+1)*mini_batch_size]], dtype=np.uint8)

                        if ims.shape[0] < mini_batch_size:
                            valid = False
                        
                        if not with_labels:
                            accumulate_batch.append(ims)
                        else:
                            accumulate_batch.append((v['aus'][(i)*mini_batch_size:(i+1)*mini_batch_size], ims))
                    if valid:
                        batches.append(tuple(accumulate_batch))
                    else:
                        print 'Ignore this batch. Patch dimensions do not coincide.'
        return batches

    def n_samples_fold(self, fold, augm):
        n = 0
        with h5py.File(self.path+'disfa.h5', 'r') as hf:
            for subject in self.get_subjects(fold):
                for k,v in hf['pose0/'+'subject_'+subject].items():
                    n += v['faces'].shape[0]
        if augm:
            return 4*n*self.n_patches
        else:
            return n*self.n_patches
    
class GeneratorFera(Generator):
    def __init__(self, path, db, type):
        Generator.__init__(self, path, db, type)

    def generate(self, partition, batch_size=32):
        segments = self._get_segments(partition)
        np.random.shuffle(segments)

        while True:
            for s in segments:
                print s
                for b in self._get_batches_meta(s, batch_size):
                    yield b
                    
    def n_samples_pose(self, pose='pose6'):
        n_train, n_test = (0,0)
        with h5py.File(self.path+'fera17.h5', 'r') as hf:
            for k,v in hf['train/'+pose].items():
                n_train += v['faces'].shape[0]
            for k,v in hf['test/'+pose].items():
                n_test += v['faces'].shape[0]
        return n_train, n_test

    def n_samples_train(self, partition):
        n=0
        if partition=='all':
            for pose in ['pose5', 'pose6']:
                n += self.n_samples_pose(pose)[0]
        else:
            n = self.n_samples_pose(partition)[0]
        return n

    def n_samples_validation(self, partition):
        n=0
        if partition=='all':
            for pose in ['pose5', 'pose6']:
                n += self.n_samples_pose(pose)[1]
        else:
            n = self.n_samples_pose(partition)[1]
        return n

'''
class GeneratorDisfa(Generator):
    def __init__(self, path, db, type):
        Generator.__init__(self, path, db, type)

    def n_samples_pose(self, pose='pose0'):
        n_train, n_test = (0,0)
        with h5py.File(self.path+'disfa.h5', 'r') as hf:
            for k,v in hf['train/'+pose].items():
                n_train += v['faces'].shape[0]
            for k,v in hf['test/'+pose].items():
                n_test += v['faces'].shape[0]
        return n_train, n_test

    def n_samples_train(self, partition):
        n=0
        if partition=='all':
            for pose in ['pose0', 'pose1']:
                n += self.n_samples_pose(pose)[0]
        else:
            n = self.n_samples_pose(partition)[0]
        return n

    def n_samples_validation(self, partition):
        n=0
        if partition=='all':
            for pose in ['pose0', 'pose1']:
                n += self.n_samples_pose(pose)[1]
        else:
            n =self.n_samples_pose(partition)[1]
        return n
'''

class GeneratorCKPlus():
    def __init__(self, path, n_channels=1, data_format='channels_last'):
        self.path = path
        self.data_format = data_format
        self.n_channels = n_channels
        (self.x_train, self.y_train) = cPickle.load(open(self.path+'ckp_train.pkl', 'rb'))
        (self.x_test, self.y_test) = cPickle.load(open(self.path+'ckp_test.pkl', 'rb'))

    def generate(self, datagen, mini_batch_size=32):
        x, y = cPickle.load(open(self.path+'ckp_train.pkl', 'rb'))

        # Repeat channels if necessary
        self.x_train = np.repeat(np.expand_dims(self.x_train, axis=3), self.n_channels, axis=3)

        if self.data_format == 'channels_first':
            self.x_train = np.rollaxis(self.x_train, 3, 1)

        while True:
            # Yield batches infinitely
            for i, (x_batch, y_batch) in enumerate(datagen.flow(x, y, batch_size=mini_batch_size)):
                yield (x_batch, y_batch)

    def load_validation(self):
        if self.data_format == 'channels_first':
            self.x_test = np.rollaxis(self.x_test, 3, 1)

        return (self.x_test, self.y_test)

    def n_iterations(self, augment, batch_size):
        return augment*self.x_train.shape[0]//batch_size

    def image_shape(self):
        if self.data_format == 'channels_last':
            return self.x_train.shape[1:]
        elif self.data_format == 'channels_first':
            return np.rollaxis(self.x_train, 3, 1).shape[1:]

    def representative_sample(self):
        if self.data_format == 'channels_last':
            return self.x_train
        elif self.data_format == 'channels_first':
            return np.rollaxis(self.x_train, 3, 1)

class GeneratorMLMNIST():
    def __init__(self, path, data_format='channels_last'):
        self.path = path
        self.data_format = data_format
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cPickle.load(open(self.path+'ml_mnist.pkl', 'rb'))

    def generate(self, datagen, mini_batch_size=32):
        if self.data_format == 'channels_first':
            self.x_train = np.rollaxis(self.x_train, 3, 1)

        while True:
            # Yield batches infinitely
            for i, (x_batch, y_batch) in enumerate(datagen.flow(self.x_train, self.y_train, batch_size=mini_batch_size)):
                yield (x_batch, y_batch)

    def load_validation(self):
        if self.data_format == 'channels_first':
            self.x_test = np.rollaxis(self.x_test, 3, 1)

        return (self.x_test, self.y_test)

    def n_iterations(self, augment, batch_size):
        return augment*self.x_train.shape[0]//batch_size

    def image_shape(self):
        if self.data_format == 'channels_last':
            return self.x_train.shape[1:]
        elif self.data_format == 'channels_first':
            return np.rollaxis(self.x_train, 3, 1).shape[1:]

    def representative_sample(self):
        if self.data_format == 'channels_last':
            return self.x_train
        elif self.data_format == 'channels_first':
            return np.rollaxis(self.x_train, 3, 1)

if __name__ == '__main__':
    path = '/Users/cipriancorneanu/Research/data/'
    path_server = '/data/data1/datasets/fera2017/'
    
    dtg = GeneratorPredsLC(path)

    for i, ((f_gt, f_pred), (m_gt, m_pred)) in enumerate(dtg.generate(partition='test', type=['faces', 'mouth'])):
        print '{}: {}'.format(i, f_gt.shape)
