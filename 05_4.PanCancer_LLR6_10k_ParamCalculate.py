import sys
import time
import pandas as pd
import numpy as np
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import copy
from scipy import stats
from sklearn import metrics

def performance_calculator(y_true, y_pred):
    # Calculate AUC
    auc = metrics.roc_auc_score(y_true, y_pred)
    # Calculate PR-AUC
    pr_auc = metrics.average_precision_score(y_true, y_pred)
    # Calculate F1 score
    f1_score = metrics.f1_score(y_true, y_pred)
    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    # Calculate geometric mean
    performance = (auc * pr_auc * f1_score * accuracy)**(1/4)
    return performance


if __name__ == "__main__":
    start_time = time.time()

    CPU_num = int(sys.argv[1])  # -1
    randomSeed = int(sys.argv[2]) # 1
    resampleNUM = int(sys.argv[3]) # 10000
    train_size = float(sys.argv[4]) # 0.8

    phenoNA = 'Response'
    LLRmodelNA = 'LLR6' # 'LR16'   'LLR6'   'LLR5noTMB'   'LLR5noChemo'
    train_sample_size_used = int(sys.argv[5]) # use how many samples to train the LLR6

    if LLRmodelNA == 'LR16':
        featuresNA6_LR = ['TMB', 'Chemo_before_IO', 'Albumin', 'FCNA', 'NLR', 'Age', 'Drug', 'Sex', 'MSI', 'Stage',
                      'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']
    if LLRmodelNA == 'LLR6':
        featuresNA6_LR = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                          'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                          'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                          'CancerType14', 'CancerType15', 'CancerType16']
    elif LLRmodelNA == 'LLR5noTMB':
        featuresNA6_LR = ['Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                          'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                          'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                          'CancerType14', 'CancerType15', 'CancerType16'] # noTMB
    elif LLRmodelNA == 'LLR5noChemo':
        featuresNA6_LR = ['TMB', 'Albumin', 'NLR', 'Age', 'CancerType1',
                          'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                          'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                          'CancerType14', 'CancerType15', 'CancerType16']  # noChemo
    xy_colNAs = ['TMB', 'Chemo_before_IO', 'Albumin', 'FCNA', 'NLR', 'Age', 'Drug', 'Sex', 'MSI', 'Stage',
                  'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI', 'CancerType1',
                  'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                  'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                  'CancerType14', 'CancerType15', 'CancerType16'] + [phenoNA] + ['CancerType']

    print('Raw data processing ...')
    dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
    dataChowell_Train0 = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
    # dataChowell_Train0 = pd.read_excel(dataALL_fn, sheet_name='Chowell2018', index_col=0)
    # dataChowell_Train0 = pd.read_excel(dataALL_fn, sheet_name='Morris_new', index_col=0)
    dataChowell_Train0 = dataChowell_Train0[xy_colNAs]
    dataChowell_Train = copy.deepcopy(dataChowell_Train0)
    if train_sample_size_used > dataChowell_Train.shape[0]:
        train_sample_size_used = dataChowell_Train.shape[0]
    train_ratio_used = min(1, train_sample_size_used / dataChowell_Train.shape[0])
    if train_ratio_used < 0.99999:
        dataChowell_Train, useless = train_test_split(dataChowell_Train, test_size=1-train_ratio_used,
                      random_state=randomSeed, stratify=dataChowell_Train['CancerType'])  # stratify=None

    # truncate extreme values of features
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    dataChowell_Train['TMB'] = [c if c < TMB_upper else TMB_upper for c in dataChowell_Train['TMB']]
    dataChowell_Train['Age'] = [c if c < Age_upper else Age_upper for c in dataChowell_Train['Age']]
    dataChowell_Train['NLR'] = [c if c < NLR_upper else NLR_upper for c in dataChowell_Train['NLR']]

    print('Chowell patient number (training): ', dataChowell_Train.shape[0])
    counter = Counter(dataChowell_Train[phenoNA])  # count examples in each class
    pos_weight = counter[0] / counter[1]  # estimate scale_pos_weight value
    print('  Phenotype name: ', phenoNA)
    print('  Negative/Positive samples in training set: ', pos_weight)

    ############## 10000-replicate random data splitting for model training and evaluation ############
    LLR_params10000 = [[], [], [], [], []]  # norm_mean, norm_std, coefs, interc, p-val
    param_dict_LLR = {'solver': 'saga', 'penalty': 'elasticnet', 'l1_ratio': 1, 'class_weight': 'balanced', 'C': 0.1, 'random_state': randomSeed}

    test_size = 1 - train_size
    AUC_score_train = []
    AUC_score_test = []
    performance_test = []
    for resampling_i in range(resampleNUM):
        data_train, data_test = train_test_split(dataChowell_Train, test_size=test_size, random_state=resampling_i*randomSeed,
                                                 stratify=None)  # stratify=None
        y_train = data_train[phenoNA]
        y_test = data_test[phenoNA]
        x_train6LR = pd.DataFrame(data_train, columns=featuresNA6_LR)
        x_test6LR = pd.DataFrame(data_test, columns=featuresNA6_LR)

        scaler_sd = StandardScaler()  # StandardScaler()
        x_train6LR = scaler_sd.fit_transform(x_train6LR)
        LLR_params10000[0].append(list(scaler_sd.mean_))
        LLR_params10000[1].append(list(scaler_sd.scale_))
        x_test6LR = scaler_sd.transform(x_test6LR)

        ############# Logistic LASSO Regression model #############
        clf = linear_model.LogisticRegression(**param_dict_LLR).fit(x_train6LR, y_train)
        y_test_pred  = clf.predict(x_test6LR)
        try:
            performance_test.append(performance_calculator(y_test, y_test_pred))
        except:
            1
        LLR_params10000[2].append(list(clf.coef_[0]))
        LLR_params10000[3].append(list(clf.intercept_))

        predictions = clf.predict(x_train6LR)
        params = np.append(clf.intercept_, clf.coef_)
        newX = np.append(np.ones((len(x_train6LR), 1)), x_train6LR, axis=1)
        MSE = (sum((y_train - predictions) ** 2)) / (len(newX) - len(newX[0]))
        try:
            var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
            sd_b = np.sqrt(var_b)
            ts_b = params / sd_b
            p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]
        except:
            p_values = [1]*(len(clf.coef_[0]) + 1)
        LLR_params10000[4].append(p_values[1:]+[p_values[0]])
    performance_test_mean = np.mean(performance_test)
    performance_test_std = np.std(performance_test)
    if train_ratio_used < 0.99999:
        fnOut = open('../03.Results/6features/PanCancer/PanCancer_' + LLRmodelNA + '_TrainSize' + str(train_sample_size_used)
                     + '_10k_ParamCalculate.txt', 'w', buffering=1)
    else:
        fnOut = open('../03.Results/6features/PanCancer/PanCancer_'+LLRmodelNA+'_10k_ParamCalculate.txt', 'w',
                     buffering=1)
    for i in range(5):
        LLR_params10000[i] = list(zip(*LLR_params10000[i]))
        LLR_params10000[i] = [np.nanmean(c) for c in LLR_params10000[i]]
    print('coef     : ', [round(c,4) for c in LLR_params10000[2]])
    print('intercept: ', [round(c,4) for c in LLR_params10000[3]])
    print('p_val: ', [round(c,4) for c in LLR_params10000[4]])
    print('Train sample size: %d. Performance_mean: %.3f. Performance_std: %.3f.'%
          (dataChowell_Train.shape[0], performance_test_mean, performance_test_std))
    fnOut.write('LLR_mean\t'+'\t'.join([str(c) for c in LLR_params10000[0]])+'\n')
    fnOut.write('LLR_scale\t' + '\t'.join([str(c) for c in LLR_params10000[1]]) + '\n')
    fnOut.write('LLR_coef\t' + '\t'.join([str(c) for c in LLR_params10000[2]]) + '\n')
    fnOut.write('LLR_intercept\t' + '\t'.join([str(c) for c in LLR_params10000[3]]) + '\n')
    fnOut.write('LLR_pval\t' + '\t'.join([str(c) for c in LLR_params10000[4]]) + '\n')
    fnOut.close()

    print('All done! Time used: ', time.time() - start_time)
