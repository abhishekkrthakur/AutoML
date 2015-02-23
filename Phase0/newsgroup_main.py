"""
AutoML : Round 0

__author__ : abhishek thakur
"""

import numpy as np
from libscores import *
from sklearn import ensemble, linear_model, preprocessing, svm, multiclass
from sklearn import decomposition, metrics, cross_validation, naive_bayes
from data_io import *
np.set_printoptions(suppress=True)

train_data = data_sparse('newsgroups/newsgroups_train.data', nbr_features=61188)
test_data = data_sparse('newsgroups/newsgroups_test.data', nbr_features=61188)
valid_data = data_sparse('newsgroups/newsgroups_valid.data', nbr_features=61188)
labels = np.loadtxt('newsgroups/newsgroups_train.solution')

clf = linear_model.SGDClassifier(loss='log', n_iter = 150, alpha=0.001, penalty='elasticnet', l1_ratio = 0.001)
#LogisticRegression(C=1., intercept_scaling=1.0)

lbl = np.argmax(labels, axis = 1)

xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(train_data, lbl)
clf.fit(xtrain, ytrain)
preds = clf.predict_proba(xtest)

print preds
print pac_metric(preprocessing.LabelBinarizer().fit_transform(ytest), preds, task='multiclass.classification')

clf.fit(train_data, lbl)
test_preds = clf.predict_proba(test_data)
valid_preds = clf.predict_proba(valid_data)

np.savetxt('res/newsgroups_test_001.predict', test_preds, '%1.5f')
np.savetxt('res/newsgroups_valid_001.predict', valid_preds, '%1.5f')

