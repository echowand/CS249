import os

import pandas as pd
import numpy as np
import xgboost as xgb
import feather
import time

# df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')
# df_all = pd.read_csv('tmp/clicks_train_50_50.csv', nrows = 5000000)
t0 = time.time()
df_all = pd.read_csv('tmp/clicks_train_50_50.csv')
# df_test = feather.read_dataframe('tmp/clicks_test.feather')
print "a", time.time() - t0
df_train_0 = df_all[df_all.fold == 0].reset_index(drop=1)
df_train_1 = df_all[df_all.fold == 1].reset_index(drop=1)
del df_train_0['fold'], df_train_1['fold'], df_all

features = list(pd.read_csv('categorical_features.txt', header=None)[0])
# features = ['ad_id','ad_document_id','ad_doc_source_id','ad_doc_publisher_id','ad_doc_top_entity','ad_doc_top_topic','ad_doc_top_cat','ad_doc_publisher_id_day','ad_doc_publisher_id_hour','ad_doc_publisher_id_geo_0']

# training a small model to select best features
# first, load the data


df_train = df_train_0[:2000000].copy()
df_val = df_train_1[:1000000].copy()



# del df_train_0, df_train_1
# df_train_0 = pd.DataFrame(data=None)
# df_train_1 = pd.DataFrame(data=None)

print "b"
for f in features:
    print('loading data for %s...' % f)
    pred_0 = 'features/mtv/%s_pred_0.npy' % f
    pred_1 = 'features/mtv/%s_pred_1.npy' % f
    rank_0 = 'features/mtv/%s_pred_rank_0.npy' % f
    rank_1 = 'features/mtv/%s_pred_rank_1.npy' % f

    df_train[f] = np.load(pred_0)[:2000000]
    df_val[f] = np.load(pred_1)[:1000000]
    df_train[f + '_rank'] = np.load(rank_0)[:2000000]
    df_val[f + '_rank'] = np.load(rank_1)[:1000000]


ignore = {'display_id', 'ad_id', 'clicked'}
columns = sorted(set(df_train.columns) - ignore)

X_t = df_train[columns].values
y_t = df_train.clicked.values

X_v = df_val[columns].values
y_v = df_val.clicked.values


dtrain = xgb.DMatrix(X_t, y_t, feature_names=columns)
dval = xgb.DMatrix(X_v, y_v, feature_names=columns)

watchlist = [(dtrain, 'train'), (dval, 'val')]
del X_t, X_v, y_t, y_v


# train a small model and save only important feautures

xgb_pars = {
    'eta': 0.3,
    'gamma': 0.0,
    'max_depth': 6,
    'min_child_weight': 100,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 0.6,
    'colsample_bylevel': 1,
    'lambda': 1,
    'alpha': 0,
    'tree_method': 'approx',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 12,
    'seed': 42,
    'silent': 1
}

print "c"
model = xgb.train(xgb_pars, dtrain, num_boost_round=20, verbose_eval=1, 
    evals=watchlist)

scores = model.get_score(importance_type='gain')
for (f, s) in scores.items():
    print (f,s)

useful_features = [f for (f, s) in scores.items() if s >= 50.0]


print len(useful_features), len(scores)
# now let's put everything together in a data frame and save the result

for f in useful_features:
    if '_rank' in f:
        base_name = f[:-5]  + '_pred_rank'
    else:
        base_name = f + '_pred'

    df_train_0[f] = np.load('features/mtv/%s_0.npy' % base_name)
    df_train_1[f] = np.load('features/mtv/%s_1.npy' % base_name)
    # df_test[f] = np.load('features/mtv/%s_test.npy' % base_name)
print "Done loading mtv", time.time() - t0

# also add the doc features

# df_train_0_doc = feather.read_dataframe('features/docs_df_train_0.feather')
# df_train_1_doc = feather.read_dataframe('features/docs_df_train_1.feather')

df_train_0_doc = pd.read_csv('features/docs_df_train_0.csv')
df_train_1_doc = pd.read_csv('features/docs_df_train_1.csv')
# # df_test_doc = feather.load_dataframe('features/docs_df_test.feather')
print "Done loading doc", time.time() - t0

doc_features = ['doc_idf_dot', 'doc_idf_dot_lsa', 'doc_idf_cos',
                'doc_idf_dot_rank', 'doc_idf_dot_lsa_rank', 'doc_idf_cos_rank']

for f in doc_features:
    df_train_0[f] = df_train_0_doc[f]
    df_train_1[f] = df_train_1_doc[f]
    # df_test[f] = df_test_doc[f]


# df_train_0['doc_known_views'] = np.load('features/doc_known_views_0.npy')
# df_train_1['doc_known_views'] = np.load('features/doc_known_views_1.npy')
# df_test['doc_known_views'] = np.load('features/doc_known_views_test.npy')


# now save evertyhing
# print df_train_0
df_train_0.to_csv('tmp/mtv_df_train_0.csv', index=False)
df_train_1.to_csv('tmp/mtv_df_train_1.csv', index=False)
print "Done writing", time.time() - t0
# feather.write_dataframe(df_train_0, 'tmp/mtv_df_train_0.feather')
# feather.write_dataframe(df_train_1, 'tmp/mtv_df_train_1.feather')
# feather.write_dataframe(df_test, 'tmp/mtv_df_test.feather')
