__author__ = 'cipriancorneanu'

import cPickle
from facepp.processor.partitioner import *
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn import preprocessing

'''
Load encoded geometry from Fera2017_frontal. Train one-vs-rest multiclass SVM
'''

# Load encoded geometries for DISFA
path_fera2017 = '/Users/cipriancorneanu/Research/data/fera2017/aligned'
dt =  cPickle.load(open(path_fera2017+'fera2017_landmarks_enc.pkl', 'rb'))

# Four axes, returned as a 2-d array
au_labels = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU17', 'AU20', 'AU25', 'AU26']

# Read data
aus, geoms, slices = (dt['aus'], dt['geom'][0][:,:-4], dt['slices'])

# Filter out non labels
aus_filtered, slices_filtered = filter(aus, slices, lambda x: np.sum(x>0)>1 ) # Filter AUS

X = np.squeeze(geoms[np.concatenate(slices_filtered)])
y = aus_filtered

# Normalize
X = preprocessing.scale(X)

# Split train-test
Xtr, Xte, ytr, yte  = train_test_split(X, y, test_size=0.25)

# Get each AU
ytr = [[0 if line[i]==0 else 1 for line in ytr] for i in range(0, ytr.shape[1])]
yte = [[0 if line[i]==0 else 1 for line in yte] for i in range(0, yte.shape[1])]

# Train classifier per AU
clf = LinearSVC()
clfs = [clf.fit(Xtr, y) for y in ytr]

# Test classifier per AU
yte_pred = [clf.predict(Xte) for clf in clfs]

acc, f1 = ([], [])
for au, au_pred in zip(yte, yte_pred):
    acc.append(accuracy_score(au, au_pred))
    f1.append(f1_score(au, au_pred))

# Evaluate
print 'Accuracy per AU = {}'.format(acc)
print 'Mean accuracy = {}'.format(np.mean(acc))

print 'F1 per AU = {}'.format(f1)
print 'Mean f1_score = {}'.format(np.mean(f1))