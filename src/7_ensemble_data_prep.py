import pandas as pd
import numpy as np
import time
from tqdm import tqdm


t0 = time.time()
df_all = pd.read_csv('tmp/clicks_train_50_50.csv')
df_train_0 = df_all[df_all.fold == 0].reset_index(drop=1)
df_train_1 = df_all[df_all.fold == 1].reset_index(drop=1)
del df_train_0['fold'], df_train_1['fold'], df_all


print "a", t0 - time.time()
# read svm predictions

df_train_0['svm'] = np.load('predictions/svm_0_preds.npy')
df_train_0['svm'] = df_train_0['svm'].astype('float32')

df_train_1['svm'] = np.load('predictions/svm_1_preds.npy')
df_train_1['svm'] = df_train_1['svm'].astype('float32')



# read ftrl predictions

ftrl_0 = pd.read_csv('predictions/ftrl_pred_0.txt')
df_train_0['ftrl'] = ftrl_0.y_pred.astype('float32')

ftrl_1 = pd.read_csv('predictions/ftrl_pred_1.txt')
df_train_1['ftrl'] = ftrl_1.y_pred.astype('float32')



# read xgb predictions

df_train_0['xgb_mtv'] = np.load('predictions/xgb_mtv_pred0.npy')
df_train_1['xgb_mtv'] = np.load('predictions/xgb_mtv_pred1.npy')


# read et predictions

df_train_0['et_mtv'] = np.load('predictions/et_pred0.npy')
df_train_1['et_mtv'] = np.load('predictions/et_pred1.npy')


# read ffm predictions

df_train_0['ffm'] = np.load('predictions/ffm_0.npy')
df_train_1['ffm'] = np.load('predictions/ffm_1.npy')


# rank features
print "ranking starts", t0 - time.time()

cols_to_rank = ['svm', 'ftrl', 'xgb_mtv', 'et_mtv', 'ffm']


for f in tqdm(cols_to_rank):
    # for df in [df_train_0, df_train_1, df_test]:
    for df in [df_train_0, df_train_1]:
        df['%s_rank' % f] = df.groupby('display_id')[f].rank(method='dense', ascending=0)
        df['%s_rank' % f] = df['%s_rank' % f].astype('uint8')

print "ranking done", t0 - time.time()
# some mean target value features

mtv_features = ['ad_document_id_on_doc_publisher_id',
                'ad_doc_source_id_on_doc_publisher_id',
                'ad_document_id_on_doc_source_id']

for f in mtv_features:
    df_train_0[f] = np.load('features/mtv/%s_pred_0.npy' % f)
    df_train_0['%s_rank' % f] = np.load('features/mtv/%s_pred_rank_0.npy' % f)

    df_train_1[f] = np.load('features/mtv/%s_pred_1.npy' % f)
    df_train_1['%s_rank' % f] = np.load('features/mtv/%s_pred_rank_1.npy' % f)


print "saving", t0 - time.time()
# now save everything

df_train_0.to_csv('tmp/df_train_0_ensemble.csv', index=False)
df_train_1.to_csv('tmp/df_train_1_ensemble.csv', index=False)


