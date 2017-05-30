__author__ = 'cipriancorneanu'

import cPickle
import numpy as np
import gc

class GeneratorFera2017():
    def __init__(self, path, data_format='channels_last', N=15):
        self.path = path
        self.data_format = data_format
        self.N = N

    def generate(self, datagen, mini_batch_size=32):
        mega_batch = 0
        #Iterate through mega_batches infinitely
        while True:
            # Load mega batch from file
            fera = cPickle.load(open(self.path+'fera17_train_' + str(mega_batch%self.N), 'rb'))
            x, y = (fera['ims'], fera['occ'])

            if self.data_format == 'channels_first':
                x = np.rollaxis(x, 3, 1)

            # Yield as many mini_batches as it fits the mega_batch_size
            for i, (x_batch, y_batch) in enumerate(datagen.flow(x, y, batch_size=mini_batch_size)):
                if i==x.shape[0]//mini_batch_size: break
                yield (x_batch, y_batch)

            mega_batch += 1
            gc.collect()

    def n_samples(self):
        mega_batch, n = (0, 0)
        while mega_batch < self.N:
            # Load mega batch from file
            y = cPickle.load(open(self.path+'fera17_train_' + str(mega_batch), 'rb'))['occ']
            mega_batch += 1
            n += len(y)
        return n

    def n_iterations(self, augment, mini_batch_size):
        mega_batch, n = (0, 0)
        while mega_batch < self.N:
        # Load mega batch from file
            y = cPickle.load(open(self.path+'fera17_train_' + str(mega_batch), 'rb'))['occ']
            mega_batch += 1
            n += len(y)//mini_batch_size
        return augment*n

    def representative_sample(self):
        return cPickle.load(open(self.path+'fera17_train_0' , 'rb'))['ims']

    def load_validation(self):
        # Validation data
        x_test, y_test = ([],[])
        for bat in [0,3]:
            dt = cPickle.load(open(self.path+'validation/'+'fera17_' + str(bat), 'rb'))
            x_test.append(dt['ims'])
            y_test.append(dt['occ'])

        return (np.concatenate(x_test), np.concatenate(y_test))

    def image_shape(self):
        if self.data_format == 'channels_last':
            return self.representative_sample().shape[1:]
        elif self.data_format == 'channels_first':
            return np.rollaxis(self.representative_sample(), 3, 1).shape[1:]

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
