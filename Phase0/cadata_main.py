"""
AutoML : Round 0

__author__ : abhishek thakur
"""

import numpy as np
from libscores import *
from sklearn import ensemble, linear_model, preprocessing
from sklearn import decomposition, metrics, cross_validation
np.set_printoptions(suppress=True)

train_data = np.loadtxt('cadata/cadata_train.data')
test_data = np.loadtxt('cadata/cadata_test.data')
valid_data = np.loadtxt('cadata/cadata_valid.data')
feat_type = np.loadtxt('cadata/cadata_feat.type', dtype = 'S20')
labels = np.loadtxt('cadata/cadata_train.solution')

train_data = np.nan_to_num(train_data)
test_data = np.nan_to_num(test_data)
valid_data = np.nan_to_num(valid_data)

clf = ensemble.GradientBoostingRegressor(verbose = 2, n_estimators = 500, max_depth = 5)

clf.fit(train_data, labels)
test_preds = clf.predict(test_data)
valid_preds = clf.predict(valid_data)

np.savetxt('res/cadata_test_001.predict', test_preds, '%1.5f')
np.savetxt('res/cadata_valid_001.predict', valid_preds, '%1.5f')