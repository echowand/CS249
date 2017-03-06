# coding: utf-8

import csv
from tqdm import tqdm
from collections import defaultdict, Counter
from math import log

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

import feather
import os 
import pandas as pd


# display_id to document_id mapping

# display_doc_ids = []

# with open('../data/events.csv') as f:
#     reader = csv.DictReader(f)
    
#     for row in tqdm(reader):
#         doc_id = int(row['document_id'])
#         display_doc_ids.append(doc_id)


# # ad_id to document_id mapping

# ad_doc_id = {}

# with open('../data/promoted_content.csv') as f:
#     reader = csv.DictReader(f)
    
#     for row in tqdm(reader):
#         ad_id = int(row['ad_id'])
#         doc_id = int(row['document_id'])
#         ad_doc_id[ad_id] = doc_id



# reading document data

categories = defaultdict(list)

with open('../data/documents_categories.csv') as f:
    reader = csv.DictReader(f)

    for row in tqdm(reader):
        doc_id = int(row['document_id'])
        cat = 'cat_' + row['category_id']
        conf = float(row['confidence_level'])
        categories[doc_id].append((cat, conf))

entities = defaultdict(list)

with open('../data/documents_entities.csv') as f:
    reader = csv.DictReader(f)
    
    for row in tqdm(reader):
        doc_id = int(row['document_id'])
        en = 'entity_' + row['entity_id']
        conf = float(row['confidence_level'])
        entities[doc_id].append((en, conf))

topics = defaultdict(list)

with open('../data/documents_topics.csv') as f:
    reader = csv.DictReader(f)
    
    for row in tqdm(reader):
        doc_id = int(row['document_id'])
        t = 'topic_' + row['topic_id']
        conf = float(row['confidence_level'])
        topics[doc_id].append((t, conf))



doc_ids = []
doc_values = []
values_cnt = Counter()

with open('../data/documents_meta.csv') as f:
    reader = csv.DictReader(f)
    i = 0
    for row in tqdm(reader):
        # if i == 1000000:
        #     break
        # i+=1
        doc_id = int(row['document_id'])
        
        source = 'src_' + row['source_id']
        if not source:
            source = 'src_unk'

        publisher = 'pub_' + row['publisher_id']
        if not publisher: 
            publisher = 'pub_unk'

        doc_vector = [(source, 1.0), (publisher, 1.0)]
        doc_vector.extend(categories[doc_id])
        doc_vector.extend(entities[doc_id])
        doc_vector.extend(topics[doc_id])

        doc_ids.append(doc_id)
        doc_values.append(dict(doc_vector))

        values_cnt.update([n for (n, _) in doc_vector])


doc_id_to_idx = {d: i for (i, d) in enumerate(doc_ids)}

print "Done with doc_id_to_idx"
# discard infrequent and calculate idf

min_df = 5
freq = {t for (t, c) in values_cnt.items() if c >= min_df}

print "aa"
N = len(doc_ids)
log_N = log(N)

idf = {k: log_N - log(v) for (k, v) in values_cnt.items() if k in freq}

print "bb"

def discard_infreq(in_dict):
    return {k: w for (k, w) in in_dict.items() if k in freq}

def idf_transform(in_dict):
    return {k: w * idf[k] for (k, w) in in_dict.items()}

doc_values = [discard_infreq(d) for d in doc_values]
idf_doc_values = [idf_transform(d) for d in doc_values]

print "cc"


del values_cnt, idf, freq, doc_values
# vectorizing the documents 

dv = DictVectorizer(dtype=np.float32, sparse=True)
X_idf = dv.fit_transform(idf_doc_values)

print "dd"
# print dv

del dv, idf_doc_values
# del categories, entities, topics, 
del doc_ids


# lsi

svd_idf = TruncatedSVD(n_components=150, random_state=1, algorithm='arpack')

print "ee"
svd_idf.fit(X_idf)

print "pass deadline"


# display_id to document_id mapping

display_doc_ids = []

with open('../data/events.csv') as f:
    reader = csv.DictReader(f)
    
    for row in tqdm(reader):
        doc_id = int(row['document_id'])
        display_doc_ids.append(doc_id)


# ad_id to document_id mapping

ad_doc_id = {}

with open('../data/promoted_content.csv') as f:
    reader = csv.DictReader(f)
    
    for row in tqdm(reader):
        ad_id = int(row['ad_id'])
        doc_id = int(row['document_id'])
        ad_doc_id[ad_id] = doc_id

# processing data in batches

def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)
        
def chunk_dataframe(df, n):
    for i in range(0, len(df), n):
        yield df.iloc[i:i+n]


def prepare_batch(batch):
    batch = batch.reset_index(drop=1)

    display_docs = (batch.display_id - 1).apply(display_doc_ids.__getitem__)
    display_docs_idx = display_docs.apply(doc_id_to_idx.get)

    ad_docs = batch.ad_id.apply(ad_doc_id.get)
    ad_docs_idx = ad_docs.apply(doc_id_to_idx.get)

    X1 = X_idf[display_docs_idx.values]
    X2 = X_idf[ad_docs_idx.values]

    dot = X1.multiply(X2).sum(axis=1)
    batch['doc_idf_dot'] = np.asarray(dot).reshape(-1)

    X1_svd = svd_idf.transform(X1)
    X2_svd = svd_idf.transform(X2)

    batch['doc_idf_dot_lsa'] = (X1_svd * X2_svd).sum(axis=1)

    X1 = normalize(X1.astype(np.float))
    X2 = normalize(X2.astype(np.float))

    dot = X1.multiply(X2).sum(axis=1)
    batch['doc_idf_cos'] = np.asarray(dot).reshape(-1)

    return batch


df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')

delete_file_if_exists('tmp/doc_features_train.csv')
print ("a")
for batch in tqdm(chunk_dataframe(df_all, n=1000000)):
    batch = prepare_batch(batch)
    append_to_csv(batch, 'tmp/doc_features_train.csv')
print ("b")
del df_all

# df_test = feather.read_dataframe('tmp/clicks_test.feather')

# delete_file_if_exists('tmp/doc_features_test.csv')

# for batch in tqdm(chunk_dataframe(df_test, n=1000000)):
#     batch = prepare_batch(batch)
#     append_to_csv(batch, 'tmp/doc_features_test.csv')

# del df_test

del svd_idf, X_idf
