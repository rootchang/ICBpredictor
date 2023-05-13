import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from lifelines import CoxPHFitter

def AUC_calculator(y, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    return auc, thresholds[ind_max]


fontSize  = 8
plt.rcParams['font.size'] = fontSize
plt.rcParams["font.family"] = "Arial"

print('Raw data read in ...')
data_survival_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
data_survival_Train = pd.read_excel(data_survival_fn, sheet_name='Chowell2015-2017', index_col=0)
data_survival_Test1 = pd.read_excel(data_survival_fn, sheet_name='Chowell2018', index_col=0)
data_survival_Test2 = pd.read_excel(data_survival_fn, sheet_name='Morris_new', index_col=0)
data_survival_Test3 = pd.read_excel(data_survival_fn, sheet_name='Morris_new2', index_col=0)
data_survival_Test4 = pd.read_excel(data_survival_fn, sheet_name='Awad_NSCLC1', index_col=0)
data_survival_Test4['Platelets'] = data_survival_Test4['Platelets'] / 1000
data_survival_Test5 = pd.read_excel(data_survival_fn, sheet_name='Awad_NSCLC2', index_col=0)
data_survival_Test5['Platelets'] = data_survival_Test5['Platelets'] / 1000
data_survival_Test6 = pd.read_excel(data_survival_fn, sheet_name='Awad_NSCLC3', index_col=0)
data_survival_Test6['Platelets'] = data_survival_Test6['Platelets'] / 1000
data_survival_Test7 = pd.read_excel(data_survival_fn, sheet_name='Lee_NSCLC', index_col=0)
data_survival_Test9 = pd.read_excel(data_survival_fn, sheet_name='Kurzrock_panCancer', index_col=0)
data_all_raw = pd.concat([data_survival_Train,data_survival_Test1,data_survival_Test2,data_survival_Test3,
                          data_survival_Test4,data_survival_Test5,data_survival_Test6,data_survival_Test7,
                          data_survival_Test9], axis=0) # data_survival_Test8,

all_features = ['CancerType', 'TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'FCNA','Drug', 'Sex', 'MSI', 'Stage',
       'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI', 'Pack_years', 'Smoking_status', 'Histology', 'Performance_status',
       'WBC', 'Neutro', 'Lymphocytes','Monocytes', 'Eosinophils', 'MLR', 'PLR', 'ELR', 'LA','LRA', 'NMR', 'ENR',
       'PNR', 'NPAR', 'NAR', 'EMR', 'PMR', 'MPAR', 'MAR','EPAR', 'EAR', 'PAR',
       'PFS_Event', 'PFS_Months', 'OS_Event', 'OS_Months', 'Response']
data_all = data_all_raw[all_features]

################ ICB response predictive value of individual features that are present in >= 2 datasets ################
cancerTypes = data_all['CancerType'].unique().tolist()
feature_performance_all_cancers = pd.DataFrame(columns=['Feature', 'AUC_mean', 'AUC_sd', 'HR_OS_mean',
                                                        'HR_OS_sd', 'HR_PFS_mean', 'HR_PFS_sd'])
candi_featureNAs = ['TMB', 'PD-L1 TPS', 'Chemotherapy history', 'Albumin', 'NLR', 'Age', 'FCNA','Drug', 'Sex', 'MSI',
       'Stage', 'LOH in HLA-I', 'HED', 'Platelets', 'Hemoglobin', 'BMI']
candi_features = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'FCNA','Drug', 'Sex', 'MSI', 'Stage',
       'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI']
for i in range(len(candi_features)):
    ft = candi_features[i]
    feature_outcome_df = data_all[[ft] + ['CancerType', 'PFS_Event', 'PFS_Months', 'OS_Event', 'OS_Months', 'Response']]
    AUC_list_ct = []
    HR_OS_list_ct = []
    HR_PFS_list_ct = []
    for ct in cancerTypes:
        feature_outcome_df_temp = feature_outcome_df.loc[feature_outcome_df['CancerType']==ct,].dropna()
        if feature_outcome_df_temp.empty:
            continue
        y = feature_outcome_df_temp['Response']
        y_pred = feature_outcome_df_temp[ft]
        AUC,threshold = AUC_calculator(y, y_pred)
        threshold = np.median(y_pred)
        y_pred_01 = [int(c >= threshold) for c in y_pred]
        if sum(y_pred_01)==0 or sum(y_pred_01)==len(y_pred_01):
            y_pred_01 = [int(c > threshold) for c in y_pred]
        if sum(y_pred_01)==0 or sum(y_pred_01)==len(y_pred_01) or len(y_pred_01) < 10:
            continue
        feature_outcome_df_temp['Score'] = y_pred_01
        feature_OS_df_temp = feature_outcome_df_temp[['Score', 'OS_Event', 'OS_Months']]
        cph = CoxPHFitter()
        cph.fit(feature_OS_df_temp, duration_col='OS_Months', event_col='OS_Event')
        HR_OS = cph.summary['exp(coef)'][0]
        feature_PFS_df_temp = feature_outcome_df_temp[['Score', 'PFS_Event', 'PFS_Months']]
        cph.fit(feature_PFS_df_temp, duration_col='PFS_Months', event_col='PFS_Event')
        HR_PFS = cph.summary['exp(coef)'][0]

        print('%40s%30s%8.2f%8.2f%8.2f%8.2f'%(candi_featureNAs[i], ct, AUC, threshold, HR_OS, HR_PFS))
        AUC_list_ct.append(AUC)
        HR_OS_list_ct.append(HR_OS)
        HR_PFS_list_ct.append(HR_PFS)
    mean_sd_AUC = [np.mean([c for c in AUC_list_ct if c < 100]),np.std([c for c in AUC_list_ct if c < 100])]
    if mean_sd_AUC[0] < 0.5:
        print(candi_featureNAs[i], ' negatively corr with ICB OR!')
        mean_sd_AUC[0] = 1 - mean_sd_AUC[0]
    mean_sd_HR_OS = [np.mean([c for c in HR_OS_list_ct if c < 100]), np.std([c for c in HR_OS_list_ct if c < 100])]
    mean_sd_HR_PFS = [np.mean([c for c in HR_PFS_list_ct if c < 100]), np.std([c for c in HR_PFS_list_ct if c < 100])]
    feature_performance_all_cancers.loc[len(feature_performance_all_cancers.index)] = [candi_featureNAs[i]] + mean_sd_AUC + mean_sd_HR_OS + mean_sd_HR_PFS

feature_performance_sorted = feature_performance_all_cancers.sort_values('AUC_mean') # , ascending=False

####### AUC of features in predicting ICB OR
fig, ax = plt.subplots(figsize=(2, 3.2))
plt.subplots_adjust(left= 0.59, bottom=0.13, right=0.93, top=0.95)
plt.axvline(x=0.5, color='gray', linestyle='--')
y_data = range(feature_performance_sorted.shape[0])
plt.errorbar(feature_performance_sorted['AUC_mean'], y_data, xerr=feature_performance_sorted['AUC_sd'], fmt='s', ecolor='k') # fmt='none'
plt.xlim([0,1])
plt.xticks([0,0.5,1])
plt.yticks(y_data)

# set the color of the negatively correlated (with ICB OR) features blue
y_axis = plt.gca().yaxis
y_ticklabels = y_axis.get_ticklabels()
for i, ticklabel in enumerate(y_ticklabels):
    y_val = y_axis.get_ticklocs()[i]
    if y_val in [15, 11, 6, 4, 3, 2]:
        ticklabel.set_color('b')

ax.set_xticklabels(['0','0.5',''])
ax.set_yticklabels(feature_performance_sorted['Feature'])
plt.xlabel('AUC (OR)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
output_fig2 = '../03.Results/Figure1C_indFeature_predPower_AUC_pancancer.pdf' # png
plt.savefig(output_fig2, transparent = True) # , dpi=300
plt.close()

####### HR_OS of features in predicting ICB OR
fig, ax = plt.subplots(figsize=(2, 3.2))
plt.subplots_adjust(left= 0.59, bottom=0.13, right=0.93, top=0.95)
plt.axvline(x=1, color='gray', linestyle='--')
y_data = range(feature_performance_sorted.shape[0])
plt.errorbar(feature_performance_sorted['HR_OS_mean'], y_data, xerr=feature_performance_sorted['HR_OS_sd'], fmt='s', ecolor='k') # fmt='none'
plt.xlim([0,3])
plt.xticks([0,1,2,3])
plt.yticks([])
ax.set_xticklabels(['0','1','2',''])
plt.xlabel('Hazard ratio (OS)')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
output_fig2 = '../03.Results/Figure1C_indFeature_predPower_OS_pancancer.pdf' # png
plt.savefig(output_fig2, transparent = True) # , dpi=300
plt.close()

####### HR_OS of features in predicting ICB OR
fig, ax = plt.subplots(figsize=(2, 3.2))
plt.subplots_adjust(left= 0.59, bottom=0.13, right=0.93, top=0.95)
plt.axvline(x=1, color='gray', linestyle='--')
y_data = range(feature_performance_sorted.shape[0])
plt.errorbar(feature_performance_sorted['HR_PFS_mean'], y_data, xerr=feature_performance_sorted['HR_PFS_sd'], fmt='s', ecolor='k') # fmt='none'
plt.xlim([0,3])
plt.xticks([0,1,2,3])
plt.yticks([])
ax.set_xticklabels(['0','1','2',''])
plt.xlabel('Hazard ratio (PFS)')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
output_fig2 = '../03.Results/Figure1C_indFeature_predPower_PFS_pancancer.pdf' # png
plt.savefig(output_fig2, transparent = True) # , dpi=300
plt.close()
