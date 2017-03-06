# CS249
Mining Information and Social Networks (https://www.kaggle.com/c/outbrain-click-prediction)

Current Progress:

Files desciption:
0_prepare_splits.py splits the training dataset into two folds: One for training, the other for validation
----------------------------------------------------------------------------------------------------------------
1_svm_data.py Process basic features for SVM by reading through 2 files: events.csv and promoted_content.csv
ad_display_str = [uuid, document_id, platform, dow, hours, dow_hour, geo_location]

Header: display_id,ad_id,clicked,fold,ad_display_str
Eg: 1,42337,0,0,addoc_938164 campaign_5969 adv_1499 u_cb8c55702adb93 d_379743 p_3 dow_1 hour_4 dow_hour_1_4 US US_SC US_SC_519
----------------------------------------------------------------------------------------------------------------
2_train_svm.py train SVM model on features generated from 1_svm_data.py. CV on 2 fold. Use AUC, F1, etc as metrics. 
Time & Result:
building the train matrix took 35.4096m
C=0.1, took 14324.763s, auc=0.734, prec=0.600, f1=0.164
----------------------------------------------------------------------------------------------------------------

Feature generation
----------------------------------------------------------------------------------------------------------------
3_doc_similarity_features.py calculates TF-IDF similarity between the document user on and the ad document
----------------------------------------------------------------------------------------------------------------
4_categorical_data_join.py and 4_categorical_data_unwrap_columnwise.py prepare data for mean target value features calculation
----------------------------------------------------------------------------------------------------------------
4_mean_target_value.py calculates mean target value for all features from categorical_features.txt

----------------------------------------------------------------------------------------------------------------
TO DO:
Waiting for features generation to be done
Feature Selection
Train XGB model on MTV features
Train FFM model
Try Ensembling
