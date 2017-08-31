__author__ = 'cipriancorneanu'

import cPickle
import h5py
import numpy as np
import gc
from sklearn.utils import shuffle
from imgaug.imgaug import augmenters as iaa
import scipy

class Generator():
    def __init__(self, path, db, type):
        self.path = path
        self.db = db
        self.type = type

    def _get_segments(self, partition):
        if partition == 'train':
            if self.db == 'disfa':
                databases = [{'fname':'disfa/disfa.h5', 'datasets':[partition+'/pose0', partition+'/pose1']}]
            elif self.db == 'fera':
                databases = [{'fname':'fera/fera17.h5', 'datasets':[partition+'/pose5', partition+'/pose6']}]
            elif self.db == 'all':
                databases = [
                    {'fname':'disfa/disfa.h5', 'datasets':[partition+'/pose0', partition+'/pose1']},
                    {'fname':'fera/fera17.h5', 'datasets':[partition+'/pose5', partition+'/pose6']}
                ]
        elif partition == 'validation':
            if self.db == 'disfa':
                databases = [{'fname':'disfa/disfa.h5', 'datasets':['test/pose0', 'test/pose1']}]
            elif self.db == 'fera':
                databases = [{'fname':'fera/fera17.h5', 'datasets':['test/pose5', 'test/pose6']}]
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
                        np.random.shuffle(data_types)
                        for dt_type in data_types:
                            s = {'db':db['fname'], 'ds':ds, 'segm': k, 'type':dt_type}
                            segments.append(s)
        return segments

    def _get_batches(self, segment, mini_batch_size, with_labels):
        batches = []
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['segm']]
            if segment['type']=='faces_patched':
                faces_patched = np.reshape(v['faces_patched'], (-1, 224, 224, 3))
                for i in range(1, faces_patched.shape[0]/mini_batch_size):
                    if with_labels:
                        batches.append((faces_patched[(i-1)*mini_batch_size:i*mini_batch_size],
                                        v['aus'][(i-1)*mini_batch_size:i*mini_batch_size]))
                    else:
                        batches.append(faces_patched[(i-1)*mini_batch_size:i*mini_batch_size])
            else:
                for i in range(1, v[segment['type']].shape[0]/mini_batch_size):
                    if with_labels:
                        batches.append((v[segment['type']][(i-1)*mini_batch_size:i*mini_batch_size],
                                   v['aus'][(i-1)*mini_batch_size:i*mini_batch_size]))
                    else:
                        batches.append(v[segment['type']][(i-1)*mini_batch_size:i*mini_batch_size])

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

    def generate(self, partition, batch_size=32, with_labels=False):
        segments = self._get_segments(partition)
        np.random.shuffle(segments)

        while True:
            for s in segments:
                print s
                for b in self._get_batches(s, batch_size, with_labels=with_labels):
                    yield b
        
                    
    def n_samples_train(self):
        disfa = GeneratorDisfa(self.path + 'disfa/', 'disfa', self.type)
        fera = GeneratorFera(self.path + 'fera/', 'fera', self.type)

        n_samples_disfa = disfa.n_samples_train()
        n_samples_fera = fera.n_samples_train()

        if self.type in ['faces', 'faces_patched', 'leye', 'reye', 'nose', 'mouth']:
            if self.db == 'fera':
                return n_samples_fera
            if self.db == 'disfa':
                return n_samples_disfa
            if self.db == 'all':
                return (n_samples_fera+n_samples_disfa)
        if self.type == 'all':
            if self.db == 'fera':
                return 6*n_samples_fera
            if self.db == 'disfa':
                return 6*n_samples_disfa
            if self.db == 'all':
                return 6*(n_samples_fera+n_samples_disfa)

    def n_samples_validation(self):
        disfa = GeneratorDisfa(self.path + 'disfa/', 'disfa', self.type)
        fera = GeneratorFera(self.path + 'fera/', 'disfa', self.type)

        n_samples_fera = fera.n_samples_test()
        n_samples_disfa = disfa.n_samples_test()

        if self.type in ['faces', 'faces_patched', 'leye', 'reye', 'nose', 'mouth']:
            if self.db == 'fera':
                return n_samples_fera
            if self.db == 'disfa':
                return n_samples_disfa
            if self.db == 'all':
                return n_samples_disfa
        if self.type == 'all':
            if self.db == 'fera':
                return 6*n_samples_fera
            if self.db == 'disfa':
                return 6*n_samples_disfa
            if self.db == 'all':
                return 6*(n_samples_disfa)

    def n_samples_test(self):
        return 30

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
                    
    def n_samples_pose(self, pose='pose0'):
        n_train, n_test = (0,0)
        with h5py.File(self.path+'fera17.h5', 'r') as hf:
            for k,v in hf['train/'+pose].items():
                n_train += v['faces'].shape[0]
            for k,v in hf['test/'+pose].items():
                n_test += v['faces'].shape[0]
        return n_train, n_test

    def n_samples_train(self):
        n=0
        for pose in ['pose5', 'pose6']:
            n += self.n_samples_pose(pose)[0]
        return n

    def n_samples_test(self):
        n=0
        for pose in ['pose5', 'pose6']:
            n += self.n_samples_pose(pose)[1]
        return n


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

    def n_samples_train(self):
        n=0
        for pose in ['pose0', 'pose1']:
            n += self.n_samples_pose(pose)[0]
        return n

    def n_samples_test(self):
        n=0
        for pose in ['pose0', 'pose1']:
            n += self.n_samples_pose(pose)[1]
        return n


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
    path_server = '/data/data1/corneanu/'

    print '---------Fera2017---------'
    dtg_fera_faces = Generator(path_server, db='fera', type='faces')
    dtg_fera_nose = Generator(path_server, db='fera', type='nose')

    dtg_disfa_faces = Generator(path_server, db='disfa', type='faces')
    dtg_disfa_mouth = Generator(path_server, db='disfa', type='mouth')
    
    print 'Fera n samples train faces: {}'.format(dtg_fera_faces.n_samples_train())
    print 'Fera n samples val faces: {}'.format(dtg_fera_faces.n_samples_validation())

    print 'Fera n samples train nose: {}'.format(dtg_fera_nose.n_samples_train())
    print 'Fera n samples val nose: {}'.format(dtg_fera_nose.n_samples_validation())

    print 'Disfa n samples train faces: {}'.format(dtg_disfa_faces.n_samples_train())
    print 'Disfa n samples val faces: {}'.format(dtg_disfa_faces.n_samples_validation())

    print 'Disfa n samples train mouth: {}'.format(dtg_disfa_mouth.n_samples_train())
    print 'Disfa n samples val mouth: {}'.format(dtg_disfa_mouth.n_samples_validation())

    '''
    for i,(obs, lbs) in enumerate(dtg_fera_faces.generate(partition='validation', batch_size=32, with_labels=True)):
        if i==0: print '{}, {}'.format(i, obs.shape)
        if i > 500 : break
    
    for i,(obs, lbs) in enumerate(dtg_fera_nose.generate(partition='validation', batch_size=32, with_labels=True)):
        if i==0: print '{}, {}'.format(i, obs.shape)
        if i > 500 : break
    
    for i,obs in enumerate(dtg_disfa_faces.generate(partition='validation', batch_size=32, with_labels=False)):
        if i==0: print '{}, {}'.format(i, obs.shape)
        if i > 500 : break
    '''
    for i,obs in enumerate(dtg_disfa_faces.generate(partition='test', batch_size=2, with_labels=False)):
        print '{}, {}'.format(i, obs.shape)
        if i > dtg_disfa_faces.n_samples_test()/2 : break
