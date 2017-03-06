# coding: utf-8

import csv
from tqdm import tqdm
from collections import defaultdict, Counter
from math import log

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

# import feather
import os 
import pandas as pd

# now processing the features and saving them as feather files

types = dict(display_id='uint32', ad_id='uint32', clicked='uint8', fold='uint8', 
             doc_idf_dot='float32', doc_idf_dot_lsa='float32', doc_idf_cos='float32')
df_all = pd.read_csv('tmp/doc_features_train.csv', dtype=types)
print ("c")
del types['clicked'], types['fold']
# df_test = pd.read_csv('tmp/doc_features_test.csv', dtype=types)
print ("d")

df_train_0 = df_all[df_all.fold == 0].reset_index(drop=1)
df_train_1 = df_all[df_all.fold == 1].reset_index(drop=1)
del df_train_0['fold'], df_train_1['fold'], df_all

cols_to_rank = ['doc_idf_dot', 'doc_idf_dot_lsa', 'doc_idf_cos']

for f in tqdm(cols_to_rank):
# for f in cols_to_rank:
    # for df in [df_train_0, df_train_1, df_test]:
    for df in [df_train_0, df_train_1]:
        df['%s_rank' % f] = df.groupby('display_id')[f].rank(ascending=0)
        df['%s_rank' % f] = df['%s_rank' % f].astype('uint8')

print ("e")

df_train_0.to_csv('features/docs_df_train_0.csv')
df_train_1.to_csv('features/docs_df_train_1.csv')
# feather.write_dataframe(df_train_0, 'features/docs_df_train_0.feather')
# feather.write_dataframe(df_train_1, 'features/docs_df_train_1.feather')
# feather.write_dataframe(df_test, 'features/docs_df_test.feather')