__author__ = 'cipriancorneanu'

import cPickle
import h5py
import numpy as np
import gc
from sklearn.utils import shuffle
from imgaug.imgaug import augmenters as iaa

class GeneratorFera2017():
    def __init__(self, path):
        self.path = path

    def get_segments(self, partition):
        databases = [
            {'fname':'fera/fera17.h5', 'datasets':[partition+'/pose5', partition+'/pose6']},
        ]

        data_types = ['face']
        #, 'face_patched', 'leye', 'reye', 'mouth', 'nose']

        segments = []
        for db in databases:
            with h5py.File(self.path+db['fname'], 'r') as hf:
                for ds in db['datasets']:
                    for k,v in hf[ds].items():
                        np.random.shuffle(data_types)
                        for dt_type in data_types:
                            s = {'db':db['fname'], 'ds':ds, 'segm': k, 'type':dt_type}
                            segments.append(s)
        return segments

    def get_batches(self, segment, mini_batch_size):
        batches = []
        print self.path+segment['db']
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['segm']]
            for i in range(1, v['faces'].shape[0]/mini_batch_size):
                batches.append((v['faces'][(i-1)*mini_batch_size:i*mini_batch_size], v['aus'][(i-1)*mini_batch_size:i*mini_batch_size]))
        return batches

    def generate_train(self, batch_size):
        segments = self.get_segments('train')
        np.random.shuffle(segments)

        while True:
            for s in segments:
                print s
                for b in self.get_batches(s, batch_size):
                    yield b

    def generate_test(self, batch_size):
        segments = self.get_segments('test')
        np.random.shuffle(segments)

        while True:
            for s in segments:
                print s
                for b in self.get_batches(s, batch_size):
                    yield b

    def n_samples_pose(self, pose='pose0'):
        n_train, n_test = (0,0)
        with h5py.File(self.path+'fera/fera17.h5', 'r') as hf:
            for k,v in hf['train/'+pose].items():
                n_train += v['faces'].shape[0]
            '''
            for k,v in hf['test/'+pose].items():
                n_test += v['faces'].shape[0]
            '''
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

class GeneratorFaces():
    # Generates from both Fera17 and Disfa
    def __init__(self, path):
        self.path = path

    def get_segments(self):
        databases = [
            {'fname':'disfa/disfa.h5', 'datasets':['train/pose0', 'train/pose1']},
            {'fname':'fera/fera17.h5', 'datasets':['train/pose5', 'train/pose6']}
        ]

        segments = []
        for db in databases:
            with h5py.File(self.path+db['fname'], 'r') as hf:
                for ds in db['datasets']:
                    for k,v in hf[ds].items():
                        s = {'db':db['fname'], 'ds':ds, 'segm': k, 'patch':True}
                        sp = {'db':db['fname'], 'ds':ds, 'segm': k, 'patch': False}
                        segments.append(s)
                        segments.append(sp)
        return segments

    def get_batches(self, segment, mini_batch_size):
        batches = []
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['segm']]
            if segment['patch']:
                faces_patched = np.reshape(v['faces_patched'], (-1, 224, 224, 3))
                for i in range(1, v['faces'].shape[0]/mini_batch_size):
                    batches.append(faces_patched[(i-1)*mini_batch_size:i*mini_batch_size])
            else:
                for i in range(1, v['faces'].shape[0]/mini_batch_size):
                    batches.append(v['faces'][(i-1)*mini_batch_size:i*mini_batch_size])
        return batches

    def generate_train(self, batch_size=32):
        segments = self.get_segments()
        np.random.shuffle(segments)

        while True:
            for s in segments:
                print s
                for b in self.get_batches(s, batch_size):
                    yield b

    def generate_validation(self, mini_batch_size=32):
        while True:
            with h5py.File(self.path+'disfa/disfa.h5', 'r') as hf:
                for k,v in hf['test/pose0'].items():
                    for i in range(1, v['faces'].shape[0]/mini_batch_size):
                        yield (v['faces'][(i-1)*mini_batch_size:i*mini_batch_size])
            gc.collect()

    def generate_test(self, mini_batch_size=32):
        #TODO : get batches from femo
        pass

    def n_samples_train(self):
        disfa = GeneratorDisfa(self.path + 'disfa/')
        fera = GeneratorFera2017(self.path + 'fera/')
        return 2*(disfa.n_samples_train()+fera.n_samples_train())

    def n_samples_test(self):
        disfa = GeneratorDisfa(self.path + 'disfa/')
        return disfa.n_samples_test()

class GeneratorFacesAndPatches():
    # Generates from both Fera17 and Disfa
    def __init__(self, path):
        self.path = path

    def get_segments(self):
        databases = [
            {'fname':'disfa/disfa.h5', 'datasets':['train/pose0', 'train/pose1']},
            {'fname':'fera/fera17.h5', 'datasets':['train/pose5', 'train/pose6']}
        ]

        data_types = ['faces', 'faces_patched', 'leye', 'reye', 'mouth', 'nose']

        segments = []
        for db in databases:
            with h5py.File(self.path+db['fname'], 'r') as hf:
                for ds in db['datasets']:
                    for k,v in hf[ds].items():
                        np.random.shuffle(data_types)
                        for dt_type in data_types:
                            s = {'db':db['fname'], 'ds':ds, 'segm': k, 'type':dt_type}
                            segments.append(s)
        return segments

    def get_batches(self, segment, mini_batch_size):
        batches = []
        with h5py.File(self.path+segment['db'], 'r') as hf:
            v = hf[segment['ds']+'/'+segment['segm']]
            faces_patched = np.reshape(v['faces_patched'], (-1, 224, 224, 3))
            if segment['type']=='faces_patched':
                for i in range(1, faces_patched.shape[0]/mini_batch_size):
                    batches.append((faces_patched[(i-1)*mini_batch_size:i*mini_batch_size],
                                    faces_patched[(i-1)*mini_batch_size:i*mini_batch_size]))
            else:
                for i in range(1, v[segment['type']].shape[0]/mini_batch_size):
                    batches.append((v[segment['type']][(i-1)*mini_batch_size:i*mini_batch_size],
                                    v[segment['type']][(i-1)*mini_batch_size:i*mini_batch_size]))

        return batches

    def generate_train(self, batch_size=32):
        segments = self.get_segments()
        np.random.shuffle(segments)

        while True:
            for s in segments:
                print s
                for b in self.get_batches(s, batch_size):
                    yield b

    def generate_validation(self, mini_batch_size=32):
        types = ['face', 'face_patched', 'leye', 'reye', 'mouth', 'nose']
        while True:
            with h5py.File(self.path+'disfa/disfa.h5', 'r') as hf:
                for k,v in hf['test/pose0'].items():
                    for type in types:
                        for i in range(1, v[type].shape[0]/mini_batch_size):
                            yield (v[type][(i-1)*mini_batch_size:i*mini_batch_size],
                                   v[type][(i-1)*mini_batch_size:i*mini_batch_size])
            gc.collect()

    def generate_test(self, mini_batch_size=32):
        #TODO : get batches from femo
        pass

    def n_samples_train(self):
        disfa = GeneratorDisfa(self.path + 'disfa/')
        fera = GeneratorFera2017(self.path + 'fera/')
        return 6*(disfa.n_samples_train()+fera.n_samples_train())

    def n_samples_test(self):
        disfa = GeneratorDisfa(self.path + 'disfa/')
        return 6*disfa.n_samples_test()

class GeneratorDisfa():
    def __init__(self, path):
        self.path = path

    def generate_train(self, mini_batch_size=32):
        while True:
            with h5py.File(self.path+'disfa.h5', 'r') as hf:
                for k,v in hf['train/pose0'].items():
                    for i in range(1, v.shape[0]/mini_batch_size):
                        yield (v['faces'][(i-1)*mini_batch_size:i*mini_batch_size],
                               v['faces'][(i-1)*mini_batch_size:i*mini_batch_size])
            gc.collect()

    def generate_test(self, mini_batch_size=32):
        while True:
            with h5py.File(self.path+'disfa.h5', 'r') as hf:
                for k,v in hf['test'].items():
                    for i in range(1, v.shape[0]/mini_batch_size):
                        yield (v['faces'][(i-1)*mini_batch_size:i*mini_batch_size],
                               v['faces'][(i-1)*mini_batch_size:i*mini_batch_size])
            gc.collect()

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
    dtg = GeneratorFera2017(path_server)

    print dtg.n_samples_train()
    print dtg.n_samples_test()

    for i,(batch, aus) in enumerate(dtg.generate_train(32)):
        print '{}, {}, {}'.format(i, batch.shape, aus.shape)
