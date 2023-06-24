
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys


phenoNA = 'Response'
LRmodelNA = sys.argv[1] #  'LR8'   'LR6'
if LRmodelNA == 'LR8':
    featuresNA = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'Drug', 'Sex', 'CancerType1',
                  'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                  'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                  'CancerType14', 'CancerType15', 'CancerType16']
elif LRmodelNA == 'LR6':
    featuresNA = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']

xy_colNAs = featuresNA + [phenoNA]

print('Raw data processing ...')
dataChowell_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
dataChowell_Train = pd.read_excel(dataChowell_fn, sheet_name='Chowell2015-2017', index_col=0)
dataChowell_Train = dataChowell_Train[xy_colNAs]

Kfold = 5
N_repeat_KFold_paramTune = 1
randomSeed = 1
Kfold_list = RepeatedKFold(n_splits=Kfold, n_repeats=N_repeat_KFold_paramTune, random_state=randomSeed)

# truncate extreme values of features
TMB_upper = 50
Age_upper = 85
NLR_upper = 25
dataChowell_Train['TMB'] = [c if c < TMB_upper else TMB_upper for c in dataChowell_Train['TMB']]
dataChowell_Train['Age'] = [c if c < Age_upper else Age_upper for c in dataChowell_Train['Age']]
dataChowell_Train['NLR'] = [c if c < NLR_upper else NLR_upper for c in dataChowell_Train['NLR']]
x_train = pd.DataFrame(dataChowell_Train, columns=featuresNA)
x_train = StandardScaler().fit_transform(x_train)
y_train = pd.DataFrame(dataChowell_Train, columns=[phenoNA])

LLRmodelNA ='LLR6' #  'LLR6'
para_fn = '../03.Results/6features/PanCancer/PanCancer_'+LLRmodelNA+'_10k_ParamCalculate.txt'
params_data = open(para_fn, 'r').readlines()
params_dict = {}
for line in params_data:
    if 'LLR_' not in line:
        continue
    words = line.strip().split('\t')
    param_name = words[0]
    params_val = [float(c) for c in words[1:]]
    params_dict[param_name] = params_val
LR = linear_model.LogisticRegression().fit(x_train, y_train)
LR.coef_ = np.array([params_dict['LLR_coef']])
LR.intercept_ = np.array(params_dict['LLR_intercept'])

if LRmodelNA == 'LR8':
    coefs = np.array(list(LR.coef_[0][0:7]) + [np.mean(abs(LR.coef_[0][7:]))])
    feature_NAs = ['TMB', 'Systemic therapy history', 'Albumin', 'NLR', 'Age', 'Drug class', 'Sex', 'Cancer type']
elif LRmodelNA == 'LR6':
    coefs = np.array(list(LR.coef_[0][0:5]) + [np.mean(abs(LR.coef_[0][5:]))])
    feature_NAs = ['TMB', 'Systemic therapy history', 'Albumin', 'NLR', 'Age', 'Cancer type']

feature_coefs_abs = abs(coefs)
sorted_idx = np.argsort(feature_coefs_abs)
coefs = coefs[sorted_idx]
feature_coefs_abs = feature_coefs_abs[sorted_idx]
feature_NAs = np.array(feature_NAs)[sorted_idx]
pos = np.arange(sorted_idx.shape[0]) + .5
figOut = '../03.Results/FigureS3_featureImportance_'+LRmodelNA+'.pdf'
featfig = plt.figure(figsize=(3.5, 3))
featax = featfig.add_subplot(1, 1, 1)
featfig.subplots_adjust(left=0.6, bottom=0.15, right=0.95, top=0.98, wspace=0.45, hspace=0.35)
featax.barh(pos, feature_coefs_abs, align='center', color='g')

featax.set_yticks(pos)
featax.set_yticklabels(feature_NAs, color='k')
featax.set_xlabel('Feature importance', color='k')
featax.set_ylabel('Feature', color='k')
featax.set_xticks([0,0.2,0.4,0.6])
featax.set_xlim([0, 0.6])
featax.set_ylim([-0.5, max(pos) + 1])
featax.spines.right.set_visible(False)
featax.spines.top.set_visible(False)
featax.figure.savefig(figOut)