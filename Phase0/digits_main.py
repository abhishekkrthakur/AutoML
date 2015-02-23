"""
AutoML : Round 0

__author__ : abhishek thakur
"""

import numpy as np
from libscores import *
from sklearn import ensemble, linear_model, preprocessing, svm
from sklearn import decomposition, metrics, cross_validation, neighbors
np.set_printoptions(suppress=True)

train_data = np.loadtxt('digits/digits_train.data')
test_data = np.loadtxt('digits/digits_test.data')
valid_data = np.loadtxt('digits/digits_valid.data')
feat_type = np.loadtxt('digits/digits_feat.type', dtype = 'S20')
labels = np.loadtxt('digits/digits_train.solution')

train_data = np.nan_to_num(train_data)
test_data = np.nan_to_num(test_data)
valid_data = np.nan_to_num(valid_data)

pca = decomposition.PCA(n_components = 40, whiten = False)
pca.fit(train_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)
valid_data = pca.transform(valid_data)

mms = preprocessing.MinMaxScaler()
mms.fit(train_data)
train_data = mms.transform(train_data)
test_data = mms.transform(test_data)
valid_data = mms.transform(valid_data)

clf = svm.SVC(C=10, verbose = 2)

clf.fit(train_data, labels)
test_preds = clf.predict(test_data)
valid_preds = clf.predict(valid_data)

np.savetxt('res/digits_test_001.predict', test_preds, '%1.5f')
np.savetxt('res/digits_valid_001.predict', valid_preds, '%1.5f')
