"""
AutoML : Round 0

__author__ : abhishek thakur
"""

import numpy as np
from libscores import *
from sklearn import ensemble, linear_model, preprocessing
from sklearn import decomposition, metrics, cross_validation
np.set_printoptions(suppress=True)

train_data = np.loadtxt('adult/adult_train.data')
test_data = np.loadtxt('adult/adult_test.data')
valid_data = np.loadtxt('adult/adult_valid.data')
feat_type = np.loadtxt('adult/adult_feat.type', dtype = 'S20')
labels = np.loadtxt('adult/adult_train.solution')

train_data = np.nan_to_num(train_data)
test_data = np.nan_to_num(test_data)
valid_data = np.nan_to_num(valid_data)

cat_features = np.where(feat_type == 'Categorical')[0]
ohe = preprocessing.OneHotEncoder(categorical_features = cat_features)
ohe.fit(train_data)
train_data = ohe.transform(train_data).toarray()
test_data = ohe.transform(test_data).toarray()
valid_data = ohe.transform(valid_data).toarray()


clf = ensemble.RandomForestClassifier(n_jobs = -1, verbose = 2, n_estimators = 7000, random_state=42)
clf.fit(train_data, labels)
test_preds = clf.predict(test_data)
valid_preds = clf.predict(valid_data)

np.savetxt('res/adult_test_001.predict', test_preds, '%1.5f')
np.savetxt('res/adult_valid_001.predict', valid_preds, '%1.5f')



