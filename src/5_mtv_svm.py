# coding: utf-8

from time import time

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score


# building the data for train

t0 = time()



# df_train_0 = feather.read_dataframe('tmp/mtv_df_train_0.feather')
df_train_0 = pd.read_csv('tmp/mtv_df_train_0.csv')
features = sorted(set(df_train_0.columns) - {'display_id', 'clicked'})

y_0 = df_train_0.clicked.values
X_0 = df_train_0[features].values
# del df_train_0
# gc.collect()


df_train_1 = pd.read_csv('tmp/mtv_df_train_1.csv')

y_1 = df_train_1.clicked.values
X_1 = df_train_1[features].values
# del df_train_1

# gc.collect()

print "Done creating X0, X1, y0, y1", time() - t0

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

np.save('predictions/mtv_svm_1_preds.npy', y_pred)

print('Fold 0 C=%s, took %.3fs, auc=%.3f' % (C, (time() - t0), auc))


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


np.save('predictions/mtv_svm_0_preds.npy', y_pred)

print('Fold 1 C=%s, took %.3fs, auc=%.3f' % (C, (time() - t0), auc))

