import pandas as pa
import numpy as np

df_all = pa.read_csv('./tmp/svm_features_train.csv')
# print df
df_train_0 = df_all[df_all.fold == 0].reset_index(drop=1)
df_train_1 = df_all[df_all.fold == 1].reset_index(drop=1)

for i in range(0, len(df_train_1)):
	# print df_train_1.iloc[i]
	vec = df_train_1.iloc[i].ad_display_str
	print vec
	# print df_train_1.iloc[i].clicked
	# break