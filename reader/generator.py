__author__ = 'cipriancorneanu'

import cPickle
import numpy as np
import gc
from skimage.transform import resize
from imgaug.imgaug import augmenters as iaa

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
            fera = cPickle.load(open(self.path+'train/fera17_train_' + str(mega_batch%self.N), 'rb'))
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
            y = cPickle.load(open(self.path+'train/fera17_train_' + str(mega_batch), 'rb'))['occ']
            mega_batch += 1
            n += len(y)
        return n

    def n_iterations(self, augment, mini_batch_size):
        mega_batch, n = (0, 0)
        while mega_batch < self.N:
        # Load mega batch from file
            y = cPickle.load(open(self.path+'train/fera17_train_' + str(mega_batch), 'rb'))['occ']
            mega_batch += 1
            n += len(y)//mini_batch_size
        return augment*n

    def representative_sample(self):
        return cPickle.load(open(self.path+'train/fera17_train_0' , 'rb'))['ims']

    def load_train(self):
        x_train, y_train = ([], [])
        for bat in range(15):
            dt = cPickle.load(open(self.path+'train/'+'fera17_train_' + str(bat), 'rb'))
            x_train.append(dt['ims'])
            y_train.append(dt['occ'])

        return (np.concatenate(x_train), np.concatenate(y_train))

    def load_validation(self):
        x_test, y_test = ([],[])
        for bat in range(10):
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
        for bat in range(15):
            dt = cPickle.load(open(self.path+'train/'+'fera17_train_' + str(bat), 'rb'))
            y.append(dt['occ'])

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
        for mega_batch in range(2):
            print('Loading mega batch {}'.format(mega_batch))
            dt = cPickle.load(open(self.path+'train/'+'fera17_train_' + str(mega_batch), 'rb'))

            print('Augmenting mega batch {}'.format(mega_batch))

            pdf = softmax([8*max(0,0.5-x) for x in [0.20, 0.19, 0.45, 0.56, 0.59, 0.56, 0.48, 0.16, 0.35, 0.16]])
            positives = [np.where(dt['occ'][:,au]==1)[0] for au in range(10)]

            n_classes, n_choices = 10, 100
            aus = np.random.choice(n_classes, n_choices, p=pdf)
            aug_idx = np.concatenate([np.random.choice(positives[au], dt['occ'].shape[0]/n_choices) for au in aus if positives[au].size])

            # Augment selected images
            aug_ims = seq.augment_images(dt['ims'][aug_idx])

            # Recreate data
            dt['ims'] = np.concatenate((dt['ims'], aug_ims))
            dt['occ'] = np.concatenate((dt['occ'], dt['occ'][aug_idx]))
            dt['int'] = np.concatenate((dt['int'], dt['int'][aug_idx]))
            dt['poses'] = np.concatenate((dt['poses'], dt['poses'][aug_idx]))
            dt['subjects'] = np.concatenate((dt['subjects'], dt['subjects'][aug_idx]))
            dt['tasks'] = np.concatenate((dt['tasks'], dt['tasks'][aug_idx]))
            dt['geoms'] = np.concatenate((dt['geoms'], dt['geoms'][aug_idx]))

            print('Dumping augmented mega batch {}'.format(mega_batch))
            cPickle.dump(dt, open(self.path+'train/fera17_train_aug_' + str(mega_batch) + '.pkl', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

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
    path = '/home/corneanu/data/fera2017/train/'
    dtg = GeneratorFera2017(path)

    print dtg.class_imbalance()
    dtg.augment()
    print dtg.class_imbalance()

    '''
    import matplotlib.pyplot as plt

    fera = cPickle.load(open(path+'fera17_train_aug_1', 'rb'))

    for i in np.random.choice(fera['ims'].shape[0], 20):
        plt.imshow(fera['ims'][i])
        plt.savefig(path + str(i) + '.png')
    '''