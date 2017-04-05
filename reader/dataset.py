__author__ = 'cipriancorneanu'

from reader import *
from extractor import *
import os
import cPickle

class ReaderFera2017():
    def __init__(self, path):
        self.path = path
        self.path_im = path + 'cohn-kanade-images/'
        self.path_lm = path + 'Landmarks/'
        self.path_emo = path + 'Emotion/'

    def read(self, fname):
        if os.path.exists(self.path+fname):
            return cPickle.load(open(self.path+fname, 'rb'))

class ReaderCKplus():
    def __init__(self, path):
        self.path = path
        self.path_im = path + 'Images/'
        self.path_lm = path + 'Landmarks/'
        self.path_emo = path + 'Emotions/'

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
                lm_seq = np.asarray(read_folder(self.path_lm+rpath), dtype=np.float16)[:,:,::-1]

                # Extract face and resize
                S = map(list, zip(*[extract(i,l,1.05,224) for i,l in zip(im_seq, lm_seq)]))

                dt['images'].append(np.asarray(S[0], dtype=np.uint8))
                dt['landmarks'].append(np.asarray(S[1], dtype=np.float16))
                dt['emos'].append(read_folder(self.path_emo+rpath))
                dt['subjects'].append(subject)
                dt['sequences'].append(sequence)

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
            im_seq = read_avi(self.path_vidl+'LeftVideo'+subject+'_comp.avi')

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

    r_ckp = ReaderCKplus(path_ckplus)
    dt = r_ckp.read('ckp.pkl')

