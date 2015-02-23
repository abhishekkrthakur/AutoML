"""
AutoML : Round 0

__author__ : abhishek thakur
"""

import numpy as np
from libscores import *
from sklearn import ensemble, linear_model, preprocessing, svm
from sklearn import decomposition, metrics, cross_validation
from data_io import *
np.set_printoptions(suppress=True)

train_data = data_binary_sparse('dorothea/dorothea_train.data', nbr_features=100000)
test_data = data_binary_sparse('dorothea/dorothea_test.data', nbr_features=100000)
valid_data = data_binary_sparse('dorothea/dorothea_valid.data', nbr_features=100000)
labels = np.loadtxt('dorothea/dorothea_train.solution')

clf1 = svm.SVC(verbose = 2, probability = True, C = 100.)
clf2 = linear_model.LogisticRegression(C=100.)
clf3 = linear_model.LogisticRegression(C=1.)

clf1.fit(train_data, labels)
clf2.fit(train_data, labels)
clf3.fit(train_data, labels)
test_preds = (clf1.predict_proba(test_data)[:,1] + clf2.predict_proba(test_data)[:,1] + clf3.predict_proba(test_data)[:,1])/3.
valid_preds = (clf1.predict_proba(valid_data)[:,1]+ clf2.predict_proba(valid_data)[:,1] + clf3.predict_proba(valid_data)[:,1])/3.

np.savetxt('res/dorothea_test_001.predict', test_preds, '%1.5f')
np.savetxt('res/dorothea_valid_001.predict', valid_preds, '%1.5f')