# coding: utf-8

from time import time

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


# building the data for train



# Uncomment to run with all sample:
# df_all = pd.read_csv('tmp/svm_features_train.csv')
# text_vec = HashingVectorizer(dtype=np.uint8, n_features=10000000, norm=None,
#                              lowercase=False, binary=True, token_pattern='\\S+',
#                              non_negative=True)

# Uncomment to run with small sample:
df_all = pd.read_csv('tmp/svm_features_train.csv', nrows = 80000)
df_all = df_all.sample(frac=0.2).reset_index(drop=True)
print df_all[:3]
text_vec = HashingVectorizer(dtype=np.uint8, n_features=10000, norm=None,
                             lowercase=False, binary=True, token_pattern='\\S+',
                             non_negative=True)

t0 = time()
X = text_vec.transform(df_all.ad_display_str)

print('building the train matrix took %.4fm' % ((time() - t0) / 60))

t0 = time()
pickle.dump( X, open( "text_vec_mat.p", "wb" ) )
print('Dumping took %.4fs' % (time() - t0) )

fold = df_all.fold.values

X_0 = X[fold == 0]
X_1 = X[fold == 1]

y = df_all.clicked.values
y_0 = y[fold == 0]
y_1 = y[fold == 1]


# fitting the model for fold 1

C = 0.1

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, C=C, random_state=1)
svm.fit(X_0, y_0)

y_pred = svm.decision_function(X_1)
auc = roc_auc_score(y_1, y_pred)
for i in range(len(y_pred)):
	if y_pred[i] > 0:
		y_pred[i] = 1
	else:
		y_pred[i] = 0

prec = precision_score(y_1, y_pred)
f1 = f1_score(y_1, y_pred)

np.save('predictions/svm_1_preds.npy', y_pred)

print('C=%s, took %.3fs, auc=%.3f, prec=%.3f, f1=%.3f' % (C, (time() - t0), auc, prec, f1))


# fitting the model for fold 0

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, C=C, random_state=1)
svm.fit(X_1, y_1)

y_pred = svm.decision_function(X_0)
auc = roc_auc_score(y_0, y_pred)

for i in range(len(y_pred)):
	if y_pred[i] > 0:
		y_pred[i] = 1
	else:
		y_pred[i] = 0

prec = precision_score(y_0, y_pred)
f1 = f1_score(y_0, y_pred)

np.save('predictions/svm_0_preds.npy', y_pred)

print('C=%s, took %.3fs, auc=%.3f, prec=%.3f, f1=%.3f' % (C, (time() - t0), auc, prec, f1))

