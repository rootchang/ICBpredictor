import time
import sys
import pandas as pd
from collections import Counter

import utils2

if __name__ == "__main__":
    start_time = time.time()


    ############################################## 0. Parameters setting ##############################################
    MLM_list1=['RF6', 'DecisionTree', 'RandomForest', 'ComplementNaiveBayes', 'MultinomialNaiveBayes', 'GaussianNaiveBayes',
               'BernoulliNaiveBayes'] # None (#6)
    MLM_list2=['LLR6', 'LogisticRegression','GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'CatBoost', 'LightGBM',
               'SupportVectorMachineLinear','SupportVectorMachinePoly','SupportVectorMachineRadial',
               'kNearestNeighbourhood','DNN','NeuralNetwork1','NeuralNetwork2','NeuralNetwork3','NeuralNetwork4',
               'GaussianProcess','QuadraticDiscriminantAnalysis'] # StandardScaler (#18)
    MLM = sys.argv[1]
    dataset = sys.argv[2] # 'Chowell', 'DFCI'
    if MLM in MLM_list1:
        SCALE = 'None'
    elif MLM in MLM_list2:
        SCALE = 'StandardScaler'
    else:
        raise Exception('MLM not recognized!')
    try:
        randomSeed = int(sys.argv[3])
    except:
        randomSeed = 1
    CPU_num = -1
    N_repeat_KFold = 1
    info_shown = 1
    Kfold = 5 # 5
    randomSearchNumber = 10000 # 10000

    phenoNA = 'Response'
    if dataset == 'Chowell':
        model_hyperParas_fn = '../03.Results/NSCLC_Chowell_ModelParaSearchResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(
            Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
        if MLM not in ['LLR6','RF6']:
            featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'FCNA', 'NLR', 'Age', 'Drug', 'Sex', 'MSI',
                          'Stage', 'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI']
        else:
            featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
        dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
        data_train1 = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
        data_train2 = pd.read_excel(dataALL_fn, sheet_name='Chowell2018', index_col=0)
        data_train = pd.concat([data_train1,data_train2],axis=0)
        data_train = data_train.loc[data_train['CancerType']=='NSCLC',]
    elif dataset == 'DFCI':
        model_hyperParas_fn = '../03.Results/NSCLC_DFCI_ModelParaSearchResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(
            Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
        if MLM not in ['LLR6', 'RF6']:
            featuresNA = ['Sex', 'Smoking_status', 'Pack_years', 'Histology',
                          'PDL1_TPS(%)', 'TMB', 'Chemo_before_IO','Treatment_Line_of_immunotherapy',
                          'Performance_status', 'Age', 'Drug',
                          'WBC', 'Neutro', 'Lymphocytes', 'Monocytes', 'Eosinophils ', 'Platelet', 'Albumin', 'NLR', 'MLR', 'PLR',
                          'ELR', 'LA', 'LRA', 'NMR', 'ENR', 'PNR', 'NPAR', 'NAR', 'EMR', 'PMR', 'MPAR', 'MAR', 'EPAR', 'EAR', 'PAR']
        else:
            featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
        dataALL_fn = '../02.Input/MarkAwad_NSCLC_all_1227.xlsx'
        data_train = pd.read_excel(dataALL_fn, sheet_name='clean', index_col=0)
    if MLM == 'LLR6':
        MLM = 'LogisticRegression'
    elif MLM == 'RF6':
        MLM = 'RandomForest'
    xy_colNAs = featuresNA + [phenoNA]
    data_train = data_train[xy_colNAs].dropna()
    cat_features = []

    ################################################# 1. Data read in #################################################
    model_hyperParas_fh = open(model_hyperParas_fn, 'w')
    print('Raw data processing ...', file=model_hyperParas_fh)

    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    data_train['TMB'] = [c if c < TMB_upper else TMB_upper for c in data_train['TMB']]
    data_train['Age'] = [c if c < Age_upper else Age_upper for c in data_train['Age']]
    data_train['NLR'] = [c if c < NLR_upper else NLR_upper for c in data_train['NLR']]
    counter = Counter(data_train[phenoNA])  # count examples in each class
    pos_weight = counter[0] / counter[1]  # estimate scale_pos_weight value
    print('  Number of all features: ', len(featuresNA), '\n  Their names: ', featuresNA, file=model_hyperParas_fh)
    print('  Phenotype name: ', phenoNA, file=model_hyperParas_fh)
    print('  Negative/Positive samples in training set: ', pos_weight, file=model_hyperParas_fh)
    print('Data size: ', data_train.shape[0], file=model_hyperParas_fh)

    scoring_dict = 'roc_auc'
    ############################### 2. Optimal model hyperparameter combination search ################################
    search_cv = utils2.optimalHyperParaSearcher(MLM, SCALE, data_train, featuresNA, phenoNA,scoring_dict, \
        randomSeed, CPU_num, N_repeat_KFold, info_shown,Kfold,cat_features, randomSearchNumber)
    print('Best params on CV sets: ', search_cv.best_params_, file=model_hyperParas_fh)
    print('Best score on CV sets: ', search_cv.best_score_, file=model_hyperParas_fh)
    print('Hyperparameter screening done! Time used: ',time.time()-start_time, file=model_hyperParas_fh)
