# CS249 Mining Information and Social Networks 
# CTR prediction 

Kaggle Project: (https://www.kaggle.com/c/outbrain-click-prediction)

Data source: https://www.kaggle.com/c/outbrain-click-prediction/data


----------------------------------------------------------------------------------------------------------------
Overview:

1. Basic classifier (SVM, FTRL) on basic features

2. Feature engineering:
	
	- Document-wise feature construction (TF-IDF)
	- Generating categorical features
	- Feature Selection

3. Train Random Forest and Gradient Boosting tree model on Mean target value features

4. Fast Factorized Machine model

5. Ensembling with Gradient Boosting tree


----------------------------------------------------------------------------------------------------------------
Files desciption:

----------------------------------------------------------------------------------------------------------------

0_prepare_splits.py splits the training dataset into two folds: One for training, the other for validation

- `1_svm_data.py` Process basic features for SVM by reading through 2 files: events.csv and promoted_content.csv

ad_display_str = [uuid, document_id, platform, dow, hours, dow_hour, geo_location]

<!-- Header: display_id,ad_id,clicked,fold,ad_display_str -->
Eg: 1,42337,0,0,addoc_938164 campaign_5969 adv_1499 u_cb8c55702adb93 d_379743 p_3 dow_1 hour_4 dow_hour_1_4 US US_SC US_SC_519

- `2_train_svm.py` train SVM model on features generated from 1_svm_data.py. CV on 2 fold. Use AUC, F1, etc as metrics. 
Time & Result:
building the train matrix took 35.4096m
C=0.1, took 14324.763s, auc=0.734, prec=0.600, f1=0.164

- `3_doc_similarity_features.py` calculates TF-IDF similarity between the document user on and the ad document

- `4_categorical_data_join.py` and `4_categorical_data_unwrap_columnwise.py` prepare data for mean target value features calculation

- `4_mean_target_value.py` calculates mean target value for all features from categorical_features.txt

- `5_best_mtv_features_xgb.py` builds an eXtreme Gradient Boosting (XBG) on a small part of data and selects best features based on information gain

- `5_mtv_rf.py` trains Random Forest model on MTV features

- `5_mtv_xgb.py` trains XGB model on MTV features and creates leaf features to be used in FFM

- `6_1_generate_ffm_data.py` creates the input file to be read by ffmlib

- `6_2_split_ffm_to_subfolds.py` splits each fold into two subfolds (can't use the original folds because the leaf features are not transferable between folds)

- `6_3_run_ffm.sh` runs libffm for training FFM models

- `6_4_put_ffm_subfolds_together.py` puts FFM predictions from each fold/subfold together

- `7_ensemble_data_prep.py` puts all the features and model predictions together for ensembling

- `7_ensemble_xgb.py` traings the second level XGB model on top of all these features

- `8_gen_net_line.py` Generate the (display+adid) - (ad_docid) - (display+adid) network for LINE input, mapped to index

- `8_line_classifiers.py` Using the LINE embedding feature vectors (tmp/) to train other models. 
----------------------------------------------------------------------------------------------------------------
Cite: 
- https://github.com/alexeygrigorev
- https://github.com/tangjianpku/LINE
