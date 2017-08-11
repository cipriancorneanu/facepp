__author__ = 'cipriancorneanu'

import cPickle
import h5py
import numpy as np
import gc
from sklearn.utils import shuffle
from imgaug.imgaug import augmenters as iaa
'''
class GeneratorFera2017(object):
    def __init__(self, path, data_format='channels_last', nbt=15, nbv=10):
        self.path = path
        self.data_format = data_format
        self.n_batches_train = nbt
        self.n_batches_validation = nbv

    def n_samples(self):
        mega_batch, n = (0, 0)
        while mega_batch < self.n_batches_train:
            # Load mega batch from file
            y = cPickle.load(open(self.path+'train/fera17_train_' + str(mega_batch), 'rb'))['occ']
            mega_batch += 1
            n += len(y)
        return n

    def n_iterations(self, augment, mini_batch_size):
        mega_batch, n = (0, 0)
        while mega_batch < self.n_batches_train:
        # Load mega batch from file
            y = cPickle.load(open(self.path+'train/fera17_train_' + str(mega_batch), 'rb'))['occ']
            mega_batch += 1
            n += len(y)//mini_batch_size
        return augment*n

    def n_iterations_augmented(self, augment, mini_batch_size):
        mega_batch, n = (0, 0)

        while mega_batch < self.n_batches_train:
            with h5py.File(self.path+'train/'+'fera17_train_aug_'+str(mega_batch)+'.h5', 'r') as hf:
                y = hf['dt']['occ'][()]
            mega_batch += 1
            n += len(y)//mini_batch_size
        return augment*n

    def representative_sample(self):
        return cPickle.load(open(self.path+'train/fera17_train_0' , 'rb'))['ims']

    def load_train(self):
        x_train, y_train = ([], [])
        for bat in range(self.n_batches_train):
            dt = cPickle.load(open(self.path+'train/'+'fera17_train_' + str(bat), 'rb'))
            x_train.append(dt['ims'])
            y_train.append(dt['occ'])

        return (np.concatenate(x_train), np.concatenate(y_train))

    def load_augmented_train(self):
        x_train, y_train = ([], [])
        for bat in range(self.n_batches_train):
            with h5py.File(self.path+'train/'+'fera17_train_aug_'+str(bat)+'.h5', 'r') as hf:
                x_train.append(hf['dt']['ims'][()])
                y_train.append(hf['dt']['occ'][()])

        return (np.concatenate(x_train), np.concatenate(y_train))

    def load_validation(self):
        x_test, y_test = ([],[])
        for bat in range(self.n_batches_validation):
            dt = cPickle.load(open(self.path+'validation/'+'fera17_' + str(bat), 'rb'))
            x_test.append(dt['ims'])
            y_test.append(dt['occ'])

        return (np.concatenate(x_test), np.concatenate(y_test))

    def image_shape(self):
        if self.data_format == 'channels_last':
            return self.representative_sample().shape[1:]
        elif self.data_format == 'channels_first':
            return np.rollaxis(self.representative_sample(), 3, 1).shape[1:]

    def class_imbalance(self):
        y = []
        for bat in range(self.n_batches_train):
            dt = cPickle.load(open(self.path+'train/'+'fera17_' + str(bat), 'rb'))
            y.append(dt['occ'])
        y = np.concatenate(y)

        return [np.sum(y[:,x])/y.shape[0] for x in range(10)]

    def class_imbalance_augmented(self):
        y = []
        for bat in range(self.n_batches_train):
            with h5py.File(self.path+'train/'+'fera17_train_aug_'+str(bat)+'.h5', 'r') as hf:
                y.append(hf['dt']['occ'][()])
        y = np.concatenate(y)

        return [np.sum(y[:,x])/y.shape[0] for x in range(10)]

    def augment(self):
        # Define augmentation
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                sometimes(iaa.PiecewiseAffine(scale=(0, 0.02))), #move parts of the image around

                iaa.SomeOf((0, 4),
                    [
                        sometimes(iaa.Multiply((0.5, 1.5))), # change brightness of images (50-150% of original value)
                        sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
                        sometimes(iaa.Grayscale(alpha=(0.0, 1.0))),
                        sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)), # add gaussian noise to images
                        sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
                        sometimes(iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))), # emboss images
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        # Apply augmentation
        for mega_batch in range(self.n_batches_train):
            print('Loading mega batch {}'.format(mega_batch))
            dt = cPickle.load(open(self.path+'train/'+'fera17_train_' + str(mega_batch), 'rb'))

            print('Augmenting mega batch {}'.format(mega_batch))

            # Compute pdf for selacting observations to augment. Higher probability to minority classes.
            pdf = softmax([8*max(0,0.5-x) for x in [0.20, 0.19, 0.45, 0.56, 0.59, 0.56, 0.48, 0.16, 0.35, 0.16]])
            positives = [np.where(dt['occ'][:,au]==1)[0] for au in range(10)]

            # Compute indices of observations to augment
            n_classes, n_choices  = 10, 100
            aus = np.random.choice(n_classes, n_choices, p=pdf)
            aug_idx = np.concatenate([np.random.choice(positives[au], dt['occ'].shape[0]/n_choices) for au in aus if positives[au].size])

            # Augment selected images
            aug_ims = seq.augment_images(dt['ims'][aug_idx])

            # Concatenate augmented and shuffle
            dt['ims'], dt['occ'], dt['int'], dt['poses'], dt['subjects'], dt['tasks'], dt['geoms'] = shuffle(
                np.concatenate((dt['ims'], aug_ims)),
                np.concatenate((dt['occ'], dt['occ'][aug_idx])),
                np.concatenate((dt['int'], dt['int'][aug_idx])),
                np.concatenate((dt['poses'], dt['poses'][aug_idx])),
                np.concatenate((dt['subjects'], dt['subjects'][aug_idx])),
                np.concatenate((dt['tasks'], dt['tasks'][aug_idx])),
                np.concatenate((dt['geoms'], dt['geoms'][aug_idx])),
            )

            print('Dumping augmented mega batch {}'.format(mega_batch))
            file = h5py.File(self.path+'train/fera17_train_aug_' + str(mega_batch)+'.h5', 'w')
            grp = file.create_group('dt')
            for k,v in dt.items():
                grp.create_dataset(k, data=v)
'''

class GeneratorFera2017():
    def __init__(self, path):
        self.path = path

    def generate_train(self):
        pass

    def n_samples_pose(self, pose='pose0'):
        n_train, n_test = (0,0)
        with h5py.File(self.path+'fera17.h5', 'r') as hf:
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

'''
class GeneratorFera2017_SI(GeneratorFera2017):
    def __init__(self, path, data_format='channels_last', nbt=15, nbv=10):
        super(GeneratorFera2017_SI, self).__init__(path=path, data_format=data_format, nbt=nbt, nbv=nbv)

    def generate(self, datagen, mini_batch_size=32):
        mega_batch = 0
        #Iterate through mega_batches infinitely
        while True:
            with h5py.File(self.path+'train/fera17_train_aug_' + str(mega_batch%self.n_batches_train)+'.h5', 'r') as hf:
                x, y = (hf['dt']['ims'][()], hf['dt']['occ'][()])

            if self.data_format == 'channels_first':
                x = np.rollaxis(x, 3, 1)

            # Yield as many mini_batches as it fits the mega_batch_size
            for i, (x_batch, y_batch) in enumerate(datagen.flow(x, y, batch_size=mini_batch_size)):
                if i==x.shape[0]//mini_batch_size: break
                yield (x_batch, y_batch)

            mega_batch += 1
            gc.collect()

class GeneratorFera2017_MI(GeneratorFera2017):
    def __init__(self, path, data_format='channels_last', nbt=15, nbv=10):
        super(GeneratorFera2017_MI, self).__init__(path=path, data_format=data_format, nbt=ntb, nbv=nbv)

    def generate(self, datagen, mini_batch_size=32):
        # TODO: scale geoms to size before yielding

        mega_batch = 0
        #Iterate through mega_batches infinitely
        while True:
            with h5py.File(self.path+'train/fera17_train_aug_' + str(mega_batch%self.n_batches_train)+'.h5', 'r') as hf:
                x1, x2, y = (hf['dt']['ims'][()], hf['dt']['geoms'][()], hf['dt']['occ'][()])

            if self.data_format == 'channels_first':
                x1 = np.rollaxis(x1, 3, 1)

            # Randomly yield as many mini_batches as it fits the mega_batch_size without replacement
            for i in range(x1.shape[0]//mini_batch_size):
                batch = np.random.choice(x1.shape[0], mini_batch_size, replace=False)
                yield ([x1[batch], x2[batch]], y[batch])

            mega_batch += 1
            gc.collect()
'''

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
                    batches.append((faces_patched[(i-1)*mini_batch_size:i*mini_batch_size],
                                    faces_patched[(i-1)*mini_batch_size:i*mini_batch_size]))
            else:
                for i in range(1, v['faces'].shape[0]/mini_batch_size):
                    batches.append((v['faces'][(i-1)*mini_batch_size:i*mini_batch_size],
                                    v['faces'][(i-1)*mini_batch_size:i*mini_batch_size]))
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
                        yield (v['faces'][(i-1)*mini_batch_size:i*mini_batch_size],
                               v['faces'][(i-1)*mini_batch_size:i*mini_batch_size])
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
    path_local = '/Users/cipriancorneanu/Research/data/disfa/'
    dtg = GeneratorFaces(path_server)

    print dtg.n_samples_train()
    print dtg.n_samples_test()

    for i, batch in enumerate(dtg.generate_train()):
        print i
