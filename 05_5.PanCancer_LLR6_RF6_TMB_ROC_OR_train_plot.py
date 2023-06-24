import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

palette = sns.color_palette("deep") # tab10, deep, muted, pastel, bright, dark, and colorblind

################# print ROC curves on all training data (and show optimal cutoff)
fontSize  = 8
plt.rcParams['font.size'] = fontSize
plt.rcParams["font.family"] = "Arial"
plt.tick_params(axis='both', direction="out", length=2)

phenoNA = 'Response'
LLRmodelNA = 'LLR6' # 'LLR6'
if LLRmodelNA == 'LLR6':
    featuresNA_LLR = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']
elif LLRmodelNA == 'LLR5noTMB':
    featuresNA_LLR = ['Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16'] # noTMB
elif LLRmodelNA == 'LLR5noChemo':
    featuresNA_LLR = ['TMB', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']  # noChemo
featuresNA_RF6 = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType_grouped']
xy_colNAs = featuresNA_LLR + ['CancerType_grouped'] + [phenoNA]
print('Raw data processing ...')
dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
dataChowellTrain = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
y_train = dataChowellTrain[phenoNA]
x_train_LLR6 = dataChowellTrain[featuresNA_LLR]
# truncate extreme values of features
TMB_upper = 50
Age_upper = 85
NLR_upper = 25
x_train_LLR6['TMB'] = [c if c < TMB_upper else TMB_upper for c in x_train_LLR6['TMB']]
x_train_LLR6['Age'] = [c if c < Age_upper else Age_upper for c in x_train_LLR6['Age']]
x_train_LLR6['NLR'] = [c if c < NLR_upper else NLR_upper for c in x_train_LLR6['NLR']]

x_train_RF6 = dataChowellTrain[featuresNA_RF6]
x_train_TMB = dataChowellTrain['TMB']

fnIn = '../03.Results/6features/PanCancer/PanCancer_'+LLRmodelNA+'_10k_ParamCalculate.txt'
params_data = open(fnIn,'r').readlines()
params_dict = {}
for line in params_data:
    if 'LLR_' not in line:
        continue
    words = line.strip().split('\t')
    param_name = words[0]
    params_val = [float(c) for c in words[1:]]
    params_dict[param_name] = params_val

ax1 = [0] * 4
fig1, ((ax1[0], ax1[1], ax1[2], ax1[3])) = plt.subplots(1, 4, figsize=(6.5, 1.5))
fig1.subplots_adjust(left=0.08, bottom=0.25, right=0.97, top=0.96, wspace=0.5, hspace=0.35)

### LLRx
scaler_sd = StandardScaler()
scaler_sd.fit(x_train_LLR6)
scaler_sd.mean_ = np.array(params_dict['LLR_mean'])
scaler_sd.scale_ = np.array(params_dict['LLR_scale'])
x_train_scaled = pd.DataFrame(scaler_sd.transform(x_train_LLR6))
clf = linear_model.LogisticRegression().fit(x_train_scaled, y_train)
clf.coef_ = np.array([params_dict['LLR_coef']])
clf.intercept_ = np.array(params_dict['LLR_intercept'])
y_pred_train = clf.predict_proba(x_train_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
specificity_sensitivity_sum = tpr + (1 - fpr)
ind_max = np.argmax(specificity_sensitivity_sum)
if ind_max < 0.5:  # the first threshold is larger than all x values (tpr=1, fpr=1)
    ind_max = 1
opt_cutoff = thresholds[ind_max]
print('LLR opt_cutoff: ', opt_cutoff) # , specificity_sensitivity_sum[ind_max]
AUC = auc(fpr, tpr)
y_pred_01 = [int(c >= opt_cutoff) for c in y_pred_train]
tn, fp, fn, tp = metrics.confusion_matrix(y_train, y_pred_01).ravel()
OddsRatio_LLR6 = (tp / (tp + fp)) / (fn / (tn + fn))
ax1[0].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
ax1[0].plot(fpr, tpr, color= palette[0],linestyle='-', label = '%s (AUC=%.2f)' % (LLRmodelNA, AUC))
ax1[0].scatter([fpr[ind_max]], [tpr[ind_max]], color='k', marker='o', s=20)
ax1[0].text(fpr[ind_max] - 0.25, tpr[ind_max] - 0.05, "%.2f" % (opt_cutoff), color='k')

###### RF6 model
params = {'n_estimators': 900, 'min_samples_split': 20, 'min_samples_leaf': 8, 'max_depth': 8}
clf = RandomForestClassifier(random_state=1, n_jobs=-1, **params).fit(x_train_RF6, y_train)
y_pred_train = clf.predict_proba(x_train_RF6)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
specificity_sensitivity_sum = tpr + (1 - fpr)
ind_max = np.argmax(specificity_sensitivity_sum)
if ind_max < 0.5:  # the first threshold is larger than all x values (tpr=1, fpr=1)
    ind_max = 1
opt_cutoff = thresholds[ind_max]
print('RF opt_cutoff: ', opt_cutoff) # , specificity_sensitivity_sum[ind_max]
AUC = auc(fpr, tpr)
y_pred_01 = [int(c >= opt_cutoff) for c in y_pred_train]
tn, fp, fn, tp = metrics.confusion_matrix(y_train, y_pred_01).ravel()
OddsRatio_RF6 = (tp / (tp + fp)) / (fn / (tn + fn))
ax1[1].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
ax1[1].plot(fpr, tpr, color= palette[2],linestyle='-', label='RF6 (AUC=%.2f)' % (AUC))
ax1[1].scatter([fpr[ind_max]], [tpr[ind_max]], color='k', marker='o', s=20)
ax1[1].text(fpr[ind_max] - 0.25, tpr[ind_max] - 0.05, "%.2f" % (opt_cutoff), color='k')

###### TMB model
y_pred_train = x_train_RF6['TMB']
fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
specificity_sensitivity_sum = tpr + (1 - fpr)
ind_max = np.argmax(specificity_sensitivity_sum)
if ind_max < 0.5:  # the first threshold is larger than all x values (tpr=1, fpr=1)
    ind_max = 1
opt_cutoff = thresholds[ind_max]
print('TMB opt_cutoff: ', opt_cutoff) # , specificity_sensitivity_sum[ind_max]
AUC = auc(fpr, tpr)
y_pred_01 = [int(c >= opt_cutoff) for c in y_pred_train]
tn, fp, fn, tp = metrics.confusion_matrix(y_train, y_pred_01).ravel()
OddsRatio_TMB = (tp / (tp + fp)) / (fn / (tn + fn))
ax1[2].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
ax1[2].plot(fpr, tpr, color= palette[3],linestyle='-', label='TMB (AUC=%.2f)' % (AUC))
ax1[2].scatter([fpr[ind_max]], [tpr[ind_max]], color='k', marker='o', s=20)
ax1[2].text(fpr[ind_max] - 0.17, tpr[ind_max] - 0.05, "%d" % (round(opt_cutoff)), color='k')

barWidth = 0.4
ax1[3].bar(np.array([0, 1, 2]) + barWidth * 1, [OddsRatio_LLR6,OddsRatio_RF6,OddsRatio_TMB],
            color=[palette[0],palette[2],palette[3]], width=barWidth, edgecolor='k')

ax1[3].set_xlim([0, 2.7])
ax1[3].set_ylim([0, 8])
ax1[3].set_yticks([0,2,4,6,8])
ax1[3].spines['right'].set_visible(False)
ax1[3].spines['top'].set_visible(False)
ax1[3].set_ylabel('Odds ratio')
ax1[3].set_xticks([])

for i in range(3):
    ax1[i].legend(frameon=False, loc=(0.15, 0.05), handlelength=1, handletextpad=0.1, labelspacing=0.2)
    ax1[i].set_xlabel('1-specificity')
    ax1[i].set_ylabel('Sensitivity')
    ax1[i].set_xlim([-0.02, 1.02])
    ax1[i].set_ylim([-0.02, 1.02])
    ax1[i].set_yticks([0,0.5,1])
    ax1[i].set_xticks([0,0.5,1])
    ax1[i].spines['right'].set_visible(False)
    ax1[i].spines['top'].set_visible(False)


output_fig1 = '../03.Results/FigureS5AA_'+LLRmodelNA+'_RF6_TMB_ROC_train.pdf'
fig1.savefig(output_fig1)
plt.close()