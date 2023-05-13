#############################################################################################
# Random search of hyperparameters for a collection of all kinds of ML models
#############################################################################################


import copy
import sys
import pandas as pd
import numpy as np
import itertools

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, ComplementNB, MultinomialNB, CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, kernels # kernels.RBF
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from skranger.ensemble import RangerForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from collections import Counter

def AUC_calculator(y, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    return auc, thresholds[ind_max]

def dataScaler(data, featuresNA, numeric_featuresNA, scaler_type):
    data_scaled = copy.deepcopy(data)
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        raise Exception('Unrecognized scaler type of %s! Only "sd" and "mM" are accepted.' % scaler_type)
    for feature in numeric_featuresNA:
        data_scaled[feature] = scaler.fit_transform(data[[feature]])
    x = pd.DataFrame(data_scaled, columns=featuresNA)
    return x

def DecisionTree_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                          scoring_dict, searchN, params=[], returnScore=False):
    dt = DecisionTreeClassifier(criterion="gini", class_weight='balanced',random_state=randomSeed)
    if not params:
        params = {'splitter': ['best', 'random'],
                  'max_features': list(np.arange(0.1, 0.91, 0.1)), # 11
                  'max_depth': list(range(3, 11)), # 11
                  'min_samples_leaf': list(range(2, 31, 2)),
                  'min_samples_split': list(range(2, 31, 2)),
                  #'max_leaf_nodes': [],
                  'ccp_alpha': [0, 0.5, 1, 10, 100],
                  }  # Total:
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=rf, param_grid=params, cv=Kfold_list, verbose=info_shown,
    #                                scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
    #                                n_jobs=CPU_num)
    search_cv = RandomizedSearchCV(estimator=dt, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    cv=Kfold_list, verbose=info_shown, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    # rf = RandomForestClassifier(n_estimators = 1000, max_features = 4, max_depth = 5, min_samples_leaf = 20,
    #                             min_samples_split = 2, max_samples = 1, class_weight = 'balanced_subsample',
    #                             bootstrap=True, criterion="gini", random_state=randomSeed, n_jobs=CPU_num,
    #                             ).fit(x_train, y_train)
    return search_cv

def rangerRandomForest_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                          scoring_dict, cat_features, searchN, params=[], returnScore=False):
    featuresNA = list(x_train.columns)
    featureNUM = x_train.shape[1]
    cat_features_index = list(map(lambda x: featuresNA.index(x), cat_features))
    rf = RangerForestClassifier(split_rule="gini", seed=randomSeed, n_jobs=CPU_num)
    if not params:
        params = {#'replace': [True, False],
                  'n_estimators': list(np.arange(200, 2100, 200)),
                  'mtry': list(range(2, min(featureNUM,11), 1)),
                  'max_depth': list(range(3, 11)),
                  'min_node_size': list(range(2, 31, 2)),
                  #'sample_fraction': [0.2,0.4,0.6,0.8,1],
                  'respect_categorical_features': ['partition'], # 'ignore', 'order', 'partition' handles categorical variables by looking at all possible
                                            # splits (2^k - 1 possibilities are evaluated) for k-level unordered factor.
                  }  # Total: 108,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=rf, param_grid=params, cv=Kfold_list, verbose=info_shown,
    #                                scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
    #                                n_jobs=CPU_num)
    search_cv = RandomizedSearchCV(estimator=rf, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    cv=Kfold_list, verbose=info_shown, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train, categorical_features = cat_features_index)
    # rf = RangerForestClassifier(replace = True, n_estimators = 1000, mtry = 4, max_depth = 5, min_node_size = 20,
    #                             sample_fraction = 1, respect_categorical_features = 'partition',
    #                             seed=randomSeed, n_jobs=CPU_num, split_rule="gini"
    #                             ).fit(x_train, y_train, class_weights = {0:1, 1:pos_weight}, categorical_features = cat_features_index)
    return search_cv


def RandomForest_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                          scoring_dict, searchN, params=[], returnScore=False):
    rf = RandomForestClassifier(bootstrap=True, criterion="gini", class_weight='balanced_subsample',
                                random_state=randomSeed, n_jobs=CPU_num)
    if not params:
        params = {'n_estimators': list(np.arange(200, 2100, 200)),
                  'max_features': list(np.arange(0.1, 0.91, 0.1)),
                  'max_depth': list(range(3, 11)), # 11
                  'min_samples_leaf': list(range(2, 31, 2)),
                  'min_samples_split': list(range(2, 31, 2)),
                  #'max_samples': [0.5,0.6,0.7,0.8,0.9,0.999], # 0.1
                  }  # Total: 108,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=rf, param_grid=params, cv=Kfold_list, verbose=info_shown,
    #                                scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
    #                                n_jobs=CPU_num)
    search_cv = RandomizedSearchCV(estimator=rf, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    cv=Kfold_list, verbose=info_shown, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    # rf = RandomForestClassifier(n_estimators = 1000, max_features = 4, max_depth = 5, min_samples_leaf = 20,
    #                             min_samples_split = 2, max_samples = 1, class_weight = 'balanced_subsample',
    #                             bootstrap=True, criterion="gini", random_state=randomSeed, n_jobs=CPU_num,
    #                             ).fit(x_train, y_train)
    return search_cv


def GBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,
                    searchN, params=[], returnScore=False):
    gb_model = GradientBoostingClassifier(random_state=randomSeed)  # , early_stopping_rounds=15
    if not params:
        params = {
            # 'loss': ['log_loss', 'deviance', 'exponential'] # default='log_loss'
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],  # default=0.1 (0, inf),
            'n_estimators': list(np.arange(200, 2100, 200)),
            #'subsample': [0.4,0.5,0.6,0.7,0.8,0.9,0.999],  # typically [0.5,1],
            # 'criterion': ['friedman_mse', 'squared_error', 'mse'], # default='friedman_mse',
            'min_samples_split': list(range(2, 31, 2)),  # list(range(2, 61, 4)),  # default=2,
            'min_samples_leaf': list(range(2, 31, 2)),  # default=1,
            'max_depth': list(np.arange(3, 11, 1)),  # default=3,
            'max_features': list(np.arange(0.1, 0.91, 0.1)),  # default=n_features,
            # Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
            # 'tol': list(10 ** np.arange(-6, -0.9, 0.5)),  # default=1e-4
        }  # Total: 756000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator = gb_model, param_grid = params, cv = Kfold_list,
    #               verbose=info_shown, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=gb_model, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,   n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    # xgb_model = XGBClassifier(learning_rate = 1, n_estimators = 1, subsample = 1,
    #                           min_samples_split = 1, min_samples_leaf = 1, max_depth = 1, max_features = 1,
    #                           random_state=randomSeed)
    return search_cv


def AdaBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,
                      cat_features, searchN, params=[], returnScore=False):
    base_estim = DecisionTreeClassifier(max_depth=1)  # , max_features=0.06
    ABM = AdaBoostClassifier(base_estimator=base_estim, random_state=randomSeed)
    if not params:
        params = {'n_estimators': list(np.arange(200, 2001, 200)),
                  'learning_rate': [0.01, 0.05, 0.03, 0.1, 0.3, 0.5, 1],  # list(np.arange(0.001,0.01,0.001)) +
                  'algorithm': ['SAMME', 'SAMME.R']
                  }  # Total 3600
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=ABM, param_grid=params, cv=Kfold_list, verbose=info_shown,
    #                          scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,   n_jobs=CPU_num)
    search_cv = RandomizedSearchCV(estimator=ABM, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,   n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    # ABM = AdaBoostClassifier(base_estimator=base_estim, n_estimators = 1, learning_rate = 1, algorithm = 'SAMME',
    #                          random_state=randomSeed) # .fit(cat_features=cat_features)
    return search_cv


def HGBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                     info_shown, scoring_dict, cat_features, searchN, params=[], returnScore=False):
    featuresNA = list(x_train.columns)
    cat_features_index = list(map(lambda x: featuresNA.index(x), cat_features))
    hgb_model = HistGradientBoostingClassifier(categorical_features=cat_features_index, random_state=randomSeed)
    if not params:
        params = {
            'loss': ['auto'],  # default='auto',
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],  # default=0.1 (0, inf) AKA shrinkage. 1 for no shrinkage,
            #'max_leaf_nodes': list(np.arange(10, 101, 10)),
            'max_iter': list(np.arange(200, 2001, 200)),  # default=100,
            'min_samples_leaf': list(range(2, 31, 2)),  # default=1,
            'max_depth': list(np.arange(3, 11, 1)),
            'l2_regularization': [0] + list(10 ** np.arange(-4, 2.1, 1)),  # default=0,
            # 'tol': list(10 ** np.arange(-7, -0.9, 1)),  # default=1e-7,
            # 'max_bins':list(range(2, 256, 10)),  # default=255,
        }  # Total 990,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator = hgb_model, param_grid = params, cv = Kfold_list,
    #               verbose=info_shown, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=hgb_model, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,   n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    # hgb_model = make_pipeline(
    #     ordinal_encoder, HistGradientBoostingClassifier(loss = 1, learning_rate = 1, max_leaf_nodes = 1, max_iter = 1,
    #                           min_samples_leaf = 1, max_depth = 1, l2_regularization = 1,
    #                           categorical_features=categorical_mask, random_state=randomSeed)
    # )
    return search_cv


def XGBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,
                     searchN, params=[], returnScore=False):
    xgb_model = XGBClassifier(objective='binary:logistic', scale_pos_weight=pos_weight, random_state=randomSeed)  #, early_stopping_rounds=15, nthread=-1, n_jobs=CPU_num,
    if not params:
        params = {
            'min_child_weight': [1] + list(range(2, 31, 2)),
            'gamma': [0] + list(10**np.arange(-2,0.1,1)),
            'subsample': [0.5, 0.8, 1],  # typically [0.5,1],
            'colsample_bytree': [0.5, 0.8, 1],  # Subsample ratio of columns when constructing each tree
            #'colsample_bylevel': [0.2,0.4,0.6,0.8,1],  # Subsample ratio of columns for each level
            #'colsample_bynode': [0.2,0.4,0.6,0.8,1],  # Subsample ratio of columns for each split
            'max_depth': list(np.arange(3, 11, 1)),  # typically [3,10] default 6,
            'n_estimators': [100]+list(np.arange(200, 1100, 200)),  # default=100,
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],  # typically [0.01,0.3] default 0.3,
            #'reg_alpha': [1],  # [0,0.5,1,1.5,2,3] #[0, 0.5, 1] default 1,
            #'reg_lambda': [0],  # list(np.arange(0,10,2)) # default 0,
            # 'tree_method': ['auto', 'exact', 'approx', 'hist', 'gpu_hist'],
        }  # Total 1,012,500
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN, 1000)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator = xgb_model, param_grid = params, cv = Kfold_list, verbose=info_shown,
    #                scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=xgb_model, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,   n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    # xgb_model = XGBClassifier(min_child_weight = 1, subsample = 1, colsample_bytree = 1, colsample_bylevel = 1,
    #                           colsample_bynode = 1, max_depth = 1,
    #                           n_estimators = 1, learning_rate = 1, reg_alpha = 1, reg_lambda = 1,
    #                           objective='binary:logistic', scale_pos_weight=pos_weight,
    #                           n_jobs=CPU_num, random_state=randomSeed)
    return search_cv


def CatBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                      scoring_dict, cat_features, searchN, params=[], returnScore=False):
    CBM = CatBoostClassifier(class_weights=[1, pos_weight], cat_features=cat_features,
                             logging_level='Silent', random_seed=randomSeed) # , thread_count=-1
    if not params:
        params = {'depth': list(np.arange(3, 11, 1)),
                  'learning_rate': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1],
                  'iterations': [100]+list(np.arange(200, 1100, 200)),  # the same as n_estimators, default 1000
                  # 'colsample_bylevel': list(np.arange(0.2, 1.1, 0.2)), # It is not recommended to change the default value
                  # of this parameter for datasets with few (10-20) features.
                  # 'max_leaves': list(np.arange(10, 101, 10)),
                  'subsample': [0.5, 0.8, 1],
                  'reg_lambda': [0, 1, 2, 3, 5, 10],  # same as l2_leaf_reg,
                  #'max_bin': list(range(254, 256, 4)),  # default=255,
                  'min_data_in_leaf': list(range(2, 31, 2)),  # default=1,
                  'random_strength': [0, 1, 2, 3, 5, 10],
                  # bootstrap_type_list = ['Bayesian','Bernoulli','MVS'],
                  }  # Total 267,300
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=CBM, param_grid=params, cv=Kfold_list, verbose=info_shown,
    #                          scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,   n_jobs=CPU_num)
    search_cv = RandomizedSearchCV(estimator=CBM, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,   n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)  # , cat_features=cat_features
    # CBM = CatBoostClassifier(class_weights = [1, pos_weight], cat_features = cat_features,
    #                          depth = 1, learning_rate = 1, iterations = 1, colsample_bylevel = 1,
    #                          subsample = 1, reg_lambda = 1, min_data_in_leaf = 2,
    #                          logging_level='Silent', random_seed=randomSeed) # .fit(cat_features=cat_features)
    return search_cv


def LightGBM_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,
                      cat_features, searchN, params=[], returnScore=False):
    featuresNA = list(x_train.columns)
    cat_features_index = list(map(lambda x: featuresNA.index(x), cat_features))
    lgb_estimator = lgb.LGBMClassifier(scale_pos_weight=pos_weight, cat_feature=cat_features_index,
                                       random_state=randomSeed, n_jobs=CPU_num)
    if not params:
        params = {  # 'min_child_weight': [0.001],
            #'min_child_samples': list(np.arange(2, 31, 2)),  # list(range(2,41,4))
            'learning_rate': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3],  # typically [0.01,0.3],
            'max_depth': list(np.arange(3, 11, 1)),  # [3, 4, 5] typically [3,10],
            'n_estimators': list(np.arange(200, 2100, 200)),
            'num_leaves': list(np.arange(10, 101, 10)),
            # 'boosting_type': ['gbdt'],
            # 'objective': ['binary'],
            'colsample_bytree': [0.2,0.4,0.6,0.8,1],  # typically [0.5,1],
            #'subsample': [0.2,0.4,0.6,0.8,1],  # typically [0.5,1],
            #'reg_alpha': [1],  # [0,0.5,1,1.5,2,3] # default 1,
            #'reg_lambda': [0],  # list(np.arange(0,10,2)) # default 0,
            'min_data_in_leaf': list(np.arange(2, 31, 2)),
            # 'lambda_l1': [0, 1, 1.5],
            # 'lambda_l2': [0, 1],
        }  # Total 675,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator = lgb_estimator, param_grid = params, cv = Kfold_list, verbose=info_shown,
    #                          scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=lgb_estimator, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, verbose=info_shown, scoring=scoring_dict, refit = 'AUC', return_train_score=True,
                                     random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    # lgb_estimator = lgb.LGBMClassifier(scale_pos_weight=pos_weight, cat_feature=cat_features_index,
    #                         min_child_samples = 2, learning_rate=0.01, max_depth=4, n_estimators=100, num_leaves = 41,
    #                         colsample_bytree=0.6, subsample=0.4, reg_alpha=1, reg_lambda=0, min_data_in_leaf = 1,
    #                         n_jobs=CPU_num, random_state=randomSeed)
    return search_cv


def ElasticNet_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                        info_shown, scoring_dict, searchN, params=[], returnScore=False):
    eNet = linear_model.ElasticNet(random_state=randomSeed)
    if not params:
        params = {"max_iter": list(np.arange(100, 5100, 200)),
                  "alpha": list(10 ** np.arange(-4, 2.1, 0.6)),
                  "l1_ratio": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                  "fit_intercept": [True, False],
                  "tol": list(10 ** np.arange(-5, -0.9, 0.4)),
                  "selection": ['cyclic', 'random']
                  }  # Total 133,100
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(eNet, param_grid = params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
    #                               n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv = RandomizedSearchCV(eNet, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # eNet = linear_model.ElasticNet(max_iter = 50, alpha = 0.001, l1_ratio = 0.3, fit_intercept = False, tol = 0.0001,
    #                             selection = 'cyclic', random_state=randomSeed).fit(x_train, y_train)
    return search_cv


def LogisticRegression_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                info_shown, scoring_dict, searchN, params=[], returnScore=False):
    LR = linear_model.LogisticRegression(random_state=randomSeed)
    if not params:
        params = {'solver': ['saga'], #['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'l1_ratio': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],  # l1_ratio=0 is using penalty='l2', l1_ratio=1 is using
                  'max_iter': range(100,1100,100),
                  'penalty': ['elasticnet'], # l1: LASSO; l2: Ridge;
                  'C': list(10 ** np.arange(-3, 3.01, 1)),
                  #'tol': list(10 ** np.arange(-6, -0.9, 1)),
                  'class_weight': ['balanced'],
                  }  # Total 30,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(LR, param_grid = params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
    #                              n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv = RandomizedSearchCV(LR, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # LR = linear_model.LogisticRegression(solver = 'newton-cg', l1_ratio=0.1, C = 0.1,
    #       max_iter = 100, tol = 0.0001, class_weight = None, random_state=randomSeed).fit(x_train, y_train)
    return search_cv

def SupportVectorMachineLinear_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                  info_shown, scoring_dict, searchN, params=[], returnScore=False):
    SVM = SVC(probability=True)
    if not params:
        params = {'C': list(10 ** (np.arange(-5, 3.1, 0.5))),  # [0.01, 0.1, 1, 10, 100] default 1.0 # C>0.01 very slow
                  'gamma': ['scale', 'auto'] + list(10 ** (np.arange(-4, 2.1, 0.5))),  # , 1, 0.1, 0.01, 0.001, 0.0001
                  'kernel': ['linear'],  # 'linear', 'poly', 'rbf', 'sigmoid'. 'poly' very slow
                  'max_iter': [-1,100,1000],
                  'tol': list(10 ** (np.arange(-5, -0.9, 0.5))),  # default 1e-3
                  'class_weight': [None, 'balanced']
                  }  # Total 10,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = GridSearchCV(SVM, param_grid=params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
                             n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)  # probability = True -> 5-Fold
    search_cv = RandomizedSearchCV(SVM, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # SVM = SVC(probability = True, C = 50, gamma = 0.001, kernel = 0.3, tol = 0.0001,
    #          class_weight = 'balanced', random_state=randomSeed).fit(x_train, y_train)
    return search_cv


def SupportVectorMachinePoly_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                  info_shown, scoring_dict, searchN, params=[], returnScore=False):
    SVM = SVC(probability=True)
    if not params:
        params = {'C': list(10 ** (np.arange(-5, 3.1, 0.5))),  # [0.01, 0.1, 1, 10, 100] default 1.0 # C>0.01 very slow
                  'gamma': ['auto'], #['scale', 'auto'] + list(10 ** (np.arange(-4, 2.1, 0.5))),  # , 1, 0.1, 0.01, 0.001, 0.0001
                  'kernel': ['poly'],  # 'linear', 'poly', 'rbf', 'sigmoid'. 'poly' very slow
                  'max_iter': [100,1000], # -1,
                  'degree': [2,3,4], # default 3 (for poly only)
                  'tol': list(10 ** (np.arange(-5, -0.9, 0.5))),  # default 1e-3
                  'class_weight': [None, 'balanced']
                  }  # Total 20,000 +
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = GridSearchCV(SVM, param_grid=params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
                             n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)  # probability = True -> 5-Fold
    search_cv = RandomizedSearchCV(SVM, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # SVM = SVC(probability = True, C = 50, gamma = 0.001, kernel = 0.3, tol = 0.0001,
    #          class_weight = 'balanced', random_state=randomSeed).fit(x_train, y_train)
    return search_cv


def SupportVectorMachineRadial_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                  info_shown, scoring_dict, searchN, params=[], returnScore=False):
    SVM = SVC(probability=True)
    if not params:
        params = {'C': list(10 ** (np.arange(-5, 3.1, 0.5))),  # [0.01, 0.1, 1, 10, 100] default 1.0 # C>0.01 very slow
                  'gamma': ['scale', 'auto'] + list(10 ** (np.arange(-4, 2.1, 0.5))),  # , 1, 0.1, 0.01, 0.001, 0.0001
                  'kernel': ['rbf'],  # 'linear', 'poly', 'rbf', 'sigmoid'. 'poly' very slow
                  'max_iter': [-1,100,1000],
                  'tol': list(10 ** (np.arange(-5, -0.9, 0.5))),  # default 1e-3
                  'class_weight': [None, 'balanced']
                  }  # Total 10,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = GridSearchCV(SVM, param_grid=params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
                             n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)  # probability = True -> 5-Fold
    search_cv = RandomizedSearchCV(SVM, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # SVM = SVC(probability = True, C = 50, gamma = 0.001, kernel = 0.3, tol = 0.0001,
    #          class_weight = 'balanced', random_state=randomSeed).fit(x_train, y_train)
    return search_cv


def BernoulliNaiveBayes_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                                 scoring_dict, searchN, params=[], returnScore=False):
    '''
    It assumes that all our features are binary such that they take only two values. Means 0s can represent “word does
    not occur in the document” and 1s as "word occurs in the document" .
    '''
    BNB = BernoulliNB()
    if not params:
        params = {
            'alpha': list(np.logspace(3, -8, num=110)),
        }  # Total 1300
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = GridSearchCV(BNB, param_grid=params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
                             n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # BNB = BernoulliNB(alpha=1.2e-6).fit(x_train, y_train)
    return search_cv


def ComplementNaiveBayes_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                                  scoring_dict, searchN, params=[], returnScore=False):
    '''
    Complement Naive Bayes is somewhat an adaptation of the standard Multinomial Naive Bayes algorithm.
    Multinomial Naive Bayes does not perform very well on imbalanced datasets. Imbalanced datasets are datasets where
    the number of examples of some class is higher than the number of examples belonging to other classes.
    In complement Naive Bayes, instead of calculating the probability of an item belonging to a certain class,
    we calculate the probability of the item belonging to all the classes. This is the literal meaning of the word,
    complement and hence is called Complement Naive Bayes.
    '''
    CNB = ComplementNB()
    if not params:
        params = {
            'alpha': list(np.logspace(3, -8, num=110)),
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = GridSearchCV(CNB, param_grid=params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
                             n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # CNB = ComplementNB(alpha=1.2e-6).fit(x_train, y_train)
    return search_cv


def MultinomialNaiveBayes_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                                   scoring_dict, searchN, params=[], returnScore=False):
    '''
    Its is used when we have discrete data (e.g. movie ratings ranging 1 and 5 as each rating will have certain
    frequency to represent). In text learning we have the count of each word to predict the class or label.
    '''
    MNB = MultinomialNB()
    if not params:
        params = {
            'alpha': list(np.logspace(3, -8, num=110)),
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = GridSearchCV(MNB, param_grid=params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
                             n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # MNB = MultinomialNB(alpha=1.2e-6).fit(x_train, y_train)
    return search_cv


def CategoricalNaiveBayes_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                                   scoring_dict, searchN, params=[], returnScore=False):
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    CaNB = CategoricalNB()
    if not params:
        params = {
            'alpha': list(np.logspace(3, -8, num=110)),
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = GridSearchCV(CaNB, param_grid=params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
                             n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # CaNB = CategoricalNB(alpha=1.2e-6).fit(x_train, y_train)
    return search_cv


def GaussianNaiveBayes_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                info_shown, scoring_dict, searchN, params=[], returnScore=False):
    '''
    Because of the assumption of the normal distribution, Gaussian Naive Bayes is used in cases when all our features
     are continuous. For example in Iris dataset features are sepal width, petal width, sepal length, petal length.
     So its features can have different values in data set as width and length can vary. We can’t represent features
     in terms of their occurrences. This means data is continuous. Hence we use Gaussian Naive Bayes here.
    '''
    GNB = GaussianNB()
    if not params:
        params = {
            'var_smoothing': [0] + list(np.logspace(3, -8, num=110)),
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = GridSearchCV(GNB, param_grid=params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
                             n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # GNB = GaussianNB(var_smoothing=1.2e-6).fit(x_train, y_train)
    return search_cv


def kNearestNeighbourhood_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                   info_shown, scoring_dict, searchN, params=[], returnScore=False):
    KNN = KNeighborsClassifier(n_jobs=CPU_num)
    if not params:
        params = {"n_neighbors": list(range(2, 61, 2)),
                  "weights": ['uniform', 'distance'],
                  "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  "leaf_size": list(range(2, 31, 2)),
                  "p": list(np.arange(1, 11, 1)),  # Minkowski metric.
                  # p = 1 is manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
                  }  # Total 36,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(KNN, param_grid = params, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  
    #                               n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv = RandomizedSearchCV(KNN, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,  n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    # KNN = KNeighborsClassifier(n_neighbors = 57, weights = 'uniform', algorithm = 'auto', leaf_size = 2, p = 2,
    #                           n_jobs=CPU_num).fit(x_train, y_train)
    return search_cv

def kerasSequential_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                           info_shown, scoring_dict, searchN, params=[], returnScore=False):
    def create_kerasSequential_model(layers, activation):
        model = Sequential()
        for i, nodes in enumerate(layers):
            if i == 0:
                model.add(Dense(nodes, input_dim=x_train.shape[1], activation=activation))
            else:
                model.add(Dense(nodes, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
        return model
    # Create the model
    model = KerasClassifier(build_fn=create_kerasSequential_model, verbose=0)
    # Define the grid search parameters
    if not params:
        params = {
            'layers': [[16], [32, 16], [64, 32, 16]],
            'activation': ['sigmoid', 'relu'],
            'batch_size': [32, 64, 128, 256],
            'epochs': [10,50,100],
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    grid = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=real_searchN, cv=Kfold_list,
                              scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,   verbose=info_shown,
                              random_state=randomSeed, n_jobs=CPU_num)
    search_cv = grid.fit(x_train, y_train)
    return search_cv

def NeuralNetwork_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                           info_shown, scoring_dict, searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            'solver': ['lbfgs'],  # ,'sgd','lbfgs', 'adam'
            # 'learning_rate_init': [0.001], #Only used when solver=’sgd’ or ‘adam’.
            'learning_rate': ["adaptive"],  # "constant", "invscaling",
            'max_iter': [1000],  # For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs
            # (how many times each data point will be used), not the number of gradient steps.
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 21), repeat=2)] +
                                  [x for x in itertools.product(range(2, 21), repeat=3)] +
                                  [x for x in itertools.product(range(2, 21), repeat=4)],
            'activation': ['relu'],  # 'logistic', 'tanh', 'relu', 'identity'
            'alpha': list(np.logspace(-6, -1, num=6)),  # default 0.0001,
            # 'early_stopping': [False],  # , False
            # 'batch_size': [100, 200, 400, 600, 800, 1000, 1200], # will not be used if solver=lbfgs
            # 'beta_1': [0.5], # Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1).
            # 'beta_2': [0.9], # Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1).
            # 'epsilon': [1e-08],  # Value for numerical stability in adam. Only used when solver=’adam’.
            # 'n_iter_no_change': [30],
            # 'power_t': [0.5], # Exponent for inverse scaling learning rate (learning_rate='invscaling' and solver='sgd')
            # 'tol': [0.01],
        }  # Total 825,246
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=MLP, param_grid=params,
    #                            cv = Kfold_list, verbose=info_shown,
    #                            scoring=scoring_dict, return_train_score=True,
    #                             n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    verbose=info_shown, random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    # MLP = MLPClassifier(activation='relu', alpha=0.1,
    #           hidden_layer_sizes=(18, 18, 20, 18), learning_rate='adaptive',
    #           max_iter=1000, solver='lbfgs', random_state=randomSeed).fit(x_train, y_train)
    return search_cv

def NeuralNetwork1_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                           info_shown, scoring_dict, searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            'solver': ['sgd','lbfgs', 'adam'],  # ,'sgd','lbfgs', 'adam'
            #'learning_rate_init': [0.001], #Only used when solver=’sgd’ or ‘adam’.
            'learning_rate': ["constant", "invscaling", "adaptive"],  # "constant", "invscaling",
            'max_iter': [100, 200, 500, 1000],  # For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs
            ## (how many times each data point will be used), not the number of gradient steps.
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 41), repeat=1)],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],  # 'logistic', 'tanh', 'relu', 'identity'
            'alpha': list(np.logspace(-6, -1, num=6)),  # default 0.0001,
            'early_stopping': [False, True],  # , False
            #'n_iter_no_change': [20],
        }  # Total 80,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=MLP, param_grid=params,
    #                            cv = Kfold_list, verbose=info_shown,
    #                            scoring=scoring_dict, return_train_score=True,
    #                             n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    verbose=info_shown, random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    # MLP = MLPClassifier(activation='relu', alpha=0.1,
    #           hidden_layer_sizes=(18, 18, 20, 18), learning_rate='adaptive',
    #           max_iter=1000, solver='lbfgs', random_state=randomSeed).fit(x_train, y_train)
    return search_cv

def NeuralNetwork2_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                           info_shown, scoring_dict, searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            #'solver': ['sgd','lbfgs', 'adam'],  # ,'sgd','lbfgs', 'adam'
            # 'learning_rate_init': [0.001], #Only used when solver=’sgd’ or ‘adam’.
            #'learning_rate': ["constant", "invscaling", "adaptive"],  # "constant", "invscaling",
            'max_iter': [100, 200, 1000],  # For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs
            # (how many times each data point will be used), not the number of gradient steps.
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 21), repeat=2)],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],  # 'logistic', 'tanh', 'relu', 'identity'
            'alpha': list(np.logspace(-6, -1, num=6)),  # default 0.0001,
            'early_stopping': [False, True],  # , False
            #'n_iter_no_change': [20],
        }  # Total 60,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=MLP, param_grid=params,
    #                            cv = Kfold_list, verbose=info_shown,
    #                            scoring=scoring_dict, return_train_score=True,
    #                             n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    verbose=info_shown, random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    # MLP = MLPClassifier(activation='relu', alpha=0.1,
    #           hidden_layer_sizes=(18, 18, 20, 18), learning_rate='adaptive',
    #           max_iter=1000, solver='lbfgs', random_state=randomSeed).fit(x_train, y_train)
    return search_cv

def NeuralNetwork3_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                           info_shown, scoring_dict, searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            #'solver': ['sgd','lbfgs', 'adam'],  # ,'sgd','lbfgs', 'adam'
            # 'learning_rate_init': [0.001], #Only used when solver=’sgd’ or ‘adam’.
            #'learning_rate': ["constant", "invscaling", "adaptive"],  # "constant", "invscaling",
            #'max_iter': [1000],  # For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs
            # (how many times each data point will be used), not the number of gradient steps.
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 21), repeat=3)],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],  # 'logistic', 'tanh', 'relu', 'identity'
            'alpha': list(np.logspace(-6, -1, num=6)),  # default 0.0001,
            # 'early_stopping': [True],  # , False
            # 'n_iter_no_change': [20],
        }  # Total 160,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=MLP, param_grid=params,
    #                            cv = Kfold_list, verbose=info_shown,
    #                            scoring=scoring_dict, return_train_score=True,
    #                             n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    verbose=info_shown, random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    # MLP = MLPClassifier(activation='relu', alpha=0.1,
    #           hidden_layer_sizes=(18, 18, 20, 18), learning_rate='adaptive',
    #           max_iter=1000, solver='lbfgs', random_state=randomSeed).fit(x_train, y_train)
    return search_cv

def NeuralNetwork4_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            # 'solver': ['sgd','lbfgs', 'adam'],  # ,'sgd','lbfgs', 'adam'
            # 'learning_rate_init': [0.001], #Only used when solver=’sgd’ or ‘adam’.
            # 'learning_rate': ["constant", "invscaling", "adaptive"],  # "constant", "invscaling",
            #'max_iter': [1000],  # For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs
            # (how many times each data point will be used), not the number of gradient steps.
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 21), repeat=4)],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],  # 'logistic', 'tanh', 'relu', 'identity'
            'alpha': list(np.logspace(-6, -1, num=6)),  # default 0.0001,
            # 'early_stopping': [True],  # , False
            # 'n_iter_no_change': [20],
        }  # Total 3,130,000
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    # search_cv = GridSearchCV(estimator=MLP, param_grid=params,
    #                            cv = Kfold_list, verbose=info_shown,
    #                            scoring=scoring_dict, return_train_score=True,
    #                             n_jobs = CPU_num)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    verbose=info_shown, random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    # MLP = MLPClassifier(activation='relu', alpha=0.1,
    #           hidden_layer_sizes=(18, 18, 20, 18), learning_rate='adaptive',
    #           max_iter=1000, solver='lbfgs', random_state=randomSeed).fit(x_train, y_train)
    return search_cv


def GaussianProcess_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params=[], returnScore=False):
    GPC = GaussianProcessClassifier(random_state=randomSeed,n_jobs = -1)
    if not params:
        params = {
            'kernel': [None,1.0 * kernels.RBF(1.0),0.1 * kernels.RBF(0.1),10 * kernels.RBF(10)], # ,0.1 * kernels.RBF(0.1),10 * kernels.RBF(10)
            'optimizer': ['fmin_l_bfgs_b',None],
            'max_iter_predict': [100,500,1000], # ,500,1000
            'n_restarts_optimizer': [0,5,10,15,20,25,30], # ,5,10,15,20,25,30
        }  # Total 168
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=GPC, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    verbose=info_shown, random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    return search_cv

def QuadraticDiscriminantAnalysis_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params=[], returnScore=False):
    QDA = QuadraticDiscriminantAnalysis()
    if not params:
        params = {
            'reg_param': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
            'store_covariance': [True,False],
            'tol': list(10 ** np.arange(-6, -0.9, 0.5)),
        }  # Total 242
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=QDA, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, 
                                    verbose=info_shown, random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    return search_cv

def optimalHyperParaSearcher(MLM, SCALE, data, featuresNA, phenoNA, scoring_dict, randomSeed, CPU_num,
                             N_repeat_KFold, info_shown, Kfold, cat_features, searchN, params_dict_list=[], returnScore=False):
    ### prepare data.
    y = data[phenoNA]
    counter = Counter(y)  # count examples in each class
    pos_weight = counter[0] / counter[1]  # estimate scale_pos_weight value
    if SCALE == 'None':
        x = pd.DataFrame(data, columns=featuresNA)
    else:
        numeric_featuresNA = list(set(featuresNA) - set(cat_features))
        if SCALE == 'StandardScaler':
            x = dataScaler(data, featuresNA, numeric_featuresNA, 'StandardScaler')
        elif SCALE == 'MinMax':
            x = dataScaler(data, featuresNA, numeric_featuresNA, 'MinMax')
        else:
            raise Exception('Unrecognized SCALE of %s! Only "None", "StandardScaler" and "MinMax" are supported.' % SCALE)

    # print('Searching for optimal hyperparameters for KFold=%d ...' % Kfold)
    if Kfold > 1.5:
        Kfold_list = RepeatedKFold(n_splits=Kfold, n_repeats=N_repeat_KFold, random_state=randomSeed) # RepeatedStratifiedKFold
    else: # get rid of cross validation
        Kfold_list = [(slice(None), slice(None))]

    if MLM == 'DecisionTree':
        search_cv = DecisionTree_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'rangerRandomForest':
        search_cv = rangerRandomForest_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict, cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'RandomForest':
        search_cv = RandomForest_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'GBoost':
        search_cv = GBoost_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'AdaBoost':
        search_cv = AdaBoost_searcher(x, y, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'HGBoost':
        search_cv = HGBoost_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'XGBoost':
        search_cv = XGBoost_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'CatBoost':
        search_cv = CatBoost_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'LightGBM':
        search_cv = LightGBM_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'ElasticNet':
        search_cv = ElasticNet_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'LogisticRegression':
        search_cv = LogisticRegression_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list, info_shown,scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'SupportVectorMachineLinear':
        search_cv = SupportVectorMachineLinear_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'SupportVectorMachinePoly':
        search_cv = SupportVectorMachinePoly_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'SupportVectorMachineRadial':
        search_cv = SupportVectorMachineRadial_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'BernoulliNaiveBayes':
        search_cv = BernoulliNaiveBayes_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'ComplementNaiveBayes':
        search_cv = ComplementNaiveBayes_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'MultinomialNaiveBayes':
        search_cv = MultinomialNaiveBayes_searcher(x, y, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'CategoricalNaiveBayes':
        search_cv = CategoricalNaiveBayes_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'GaussianNaiveBayes':
        search_cv = GaussianNaiveBayes_searcher(x, y, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'kNearestNeighbourhood':
        search_cv = kNearestNeighbourhood_searcher(x, y, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'DNN':
        search_cv = kerasSequential_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'NeuralNetwork1':
        search_cv = NeuralNetwork1_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'NeuralNetwork2':
        search_cv = NeuralNetwork2_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'NeuralNetwork3':
        search_cv = NeuralNetwork3_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'NeuralNetwork4':
        search_cv = NeuralNetwork4_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'GaussianProcess':
        search_cv = GaussianProcess_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'QuadraticDiscriminantAnalysis':
        search_cv = QuadraticDiscriminantAnalysis_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    else:
        raise Exception('Unrecognized machine learning algorithm MLM of %s!' % MLM)
    return search_cv
