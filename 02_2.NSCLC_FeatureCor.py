import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.sandbox.stats.multicomp import multipletests

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

################ Correlation matrix heatmap plot between continuous variables that present in >= 2 datasets (NSCLC) ################
data_NSCLC = data_all_raw.loc[data_all_raw['CancerType']=='NSCLC',]
continuous_features = ['TMB', 'PDL1_TPS(%)', 'Age', 'Pack_years',
       'Albumin', 'Platelets', 'WBC', 'Neutro', 'Lymphocytes','Monocytes', 'Eosinophils', 'NLR', 'MLR', 'PLR', 'ELR', 'LA','LRA',
       'NMR', 'ENR', 'PNR', 'NPAR', 'NAR', 'EMR', 'PMR', 'MPAR', 'MAR','EPAR', 'EAR', 'PAR']
continuous_features_full = ['TMB', 'PD-L1 TPS', 'Age', 'Tobacco use',
       'Albumin', 'Platelets', 'White blood cells (WBC)', 'Neutrophils', 'Lymphocytes','Monocytes', 'Eosinophils',
       'Neutrophil / lymphocyte', 'Monocyte / lymphocyte', 'Platelet / lymphocyte',
       'Eosinophil / lymphocyte', 'Lymphocyte * albumin','Lymphocyte / WBC * albumin',
       'Neutrophil / monocyte', 'Eosinophil / neutrophil', 'Platelet / neutrophil', 'Neutrophil / WBC / albumin',
       'Neutrophil / albumin', 'Eosinophil / monocyte', 'Platelet / monocyte', 'Monocyte / WBC / albumin',
       'Monocyte / albumin','Eosinophil / WBC / albumin', 'Eosinophil / albumin', 'Platelet / albumin']
data_continuous_features = data_NSCLC[continuous_features]
corr_out = data_continuous_features.corr(method='spearman', min_periods=1)
# set column and row names
corr_out.columns = data_continuous_features.columns
corr_out.index = data_continuous_features.columns
# create a mask to only show the lower triangle
mask = np.zeros_like(corr_out)
mask[np.triu_indices_from(mask)] = True
# set heatmap color palette and breaks
palette_length = 400
my_color = sns.color_palette("RdBu_r", n_colors=palette_length)

# plot correlation heatmap
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left= 0.02, bottom=0.02, right=0.85, top=0.95)
heatmap = sns.heatmap(corr_out, mask=mask, cmap=my_color, center=0,
                vmin=-1, vmax=1, xticklabels=True, yticklabels=True,
                cbar=True, cbar_kws={"shrink": 0.5, "label": "Spearman correlation coefficient"}, cbar_ax=ax.inset_axes([1, 0.3, 0.03, 0.7]),
                linewidths=0.1, linecolor='white', square=True, ax=ax) # annot=True, fmt='.2f', annot_kws={"fontsize":7},
# Define significance levels
sig_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]
# calculate significance symbols
p_list = []
for i in range(corr_out.shape[1]):
    for j in range(corr_out.shape[1]-1,i,-1):
        if mask[i, j]:
            corr, pval = spearmanr(data_continuous_features.iloc[:,i], data_continuous_features.iloc[:,j], nan_policy='omit')
            p_list.append(pval)
# adjusting p-values with multipletests
adjusted_p_values = multipletests(p_list, method='bonferroni')[1] # bonferroni   fdr_bh
# add significance symbols
count = 0
for i in range(corr_out.shape[1]):
    for j in range(corr_out.shape[1]-1,i,-1):
        if mask[i, j]:
            anno_text = '%.2f' % corr_out.iloc[i,j]
            adj_pval = adjusted_p_values[count]
            count+=1
            for level in sig_levels:
                if adj_pval < level[0]:
                    anno_text = '%.2f\n%s' % (corr_out.iloc[i,j], level[1])
                    break
            ax.text(i + 0.5, j + 0.5, anno_text, ha='center', va='center', fontsize=7, color='k')
cbar = heatmap.collections[0].colorbar
# Set the font size and font color for the colorbar
cbar.ax.tick_params(labelsize=8)
# display the column names at the diagonal
for i in range(len(corr_out.columns)):
    plt.text(i + 0.5, i + 0.5, continuous_features_full[i], ha='left', va='bottom', rotation=45, fontsize=8)
# show the plot
plt.xticks([])
plt.yticks([])
output_fig1 = '../03.Results/FigureS1_corHeatmap_NSCLC.pdf' # png
plt.savefig(output_fig1, transparent = True) # , dpi=300
plt.close()

