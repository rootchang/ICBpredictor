import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys
import colorcet as cc

fontSize  = 10
plt.rcParams['font.size'] = fontSize
plt.rcParams["font.family"] = "Arial"

fig, axes = plt.subplots(1, 5, figsize=(6.5, 5.2))
plt.subplots_adjust(left=0.33, bottom=0.1, right=0.98, top=0.99, wspace=0.3, hspace=0.45)

palette = sns.color_palette(cc.glasbey, n_colors=19) # tab10, deep, muted, pastel, bright, dark, and colorblind

print('Raw data reading in ...')
resampleNUM = 10000
# MLM_list = ['DecisionTree', 'RandomForest', 'ComplementNaiveBayes', 'MultinomialNaiveBayes', 'GaussianNaiveBayes',
#              'BernoulliNaiveBayes', 'LogisticRegression', 'GBoost', 'AdaBoost', 'HGBoost', 'XGBoost',
#              'LightGBM', 'SupportVectorMachineLinear', 'SupportVectorMachinePoly', 'SupportVectorMachineRadial',
#              'kNearestNeighbourhood', 'DNN', 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3', 'NeuralNetwork4',
#              'GaussianProcess', 'QuadraticDiscriminantAnalysis']
MLM_list = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'GBoost', 'AdaBoost', 'HGBoost', 'XGBoost',
             'LightGBM', 'SupportVectorMachineRadial',
             'kNearestNeighbourhood', 'DNN', 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3', 'NeuralNetwork4',
             'GaussianProcess'] # 'TMB', 'RF6', 'RF16_NBT',
MLM_list1=['TMB', 'RF6', 'RF16_NBT', 'DecisionTree', 'RandomForest', 'ComplementNaiveBayes', 'MultinomialNaiveBayes',
           'GaussianNaiveBayes', 'BernoulliNaiveBayes', 'RF16_NBT'] # None (#6)
MLM_list2=['LLR6', 'LR5noTMB', 'LR5noChemo', 'LogisticRegression','GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'CatBoost', 'LightGBM',
           'SupportVectorMachineLinear','SupportVectorMachinePoly','SupportVectorMachineRadial',
           'kNearestNeighbourhood','DNN','NeuralNetwork1','NeuralNetwork2','NeuralNetwork3','NeuralNetwork4',
           'GaussianProcess','QuadraticDiscriminantAnalysis'] # StandardScaler (#18)

MLM_name_map = {'LLR6': 'LLR6', 'LR5noTMB': 'LR5(noTMB)', 'LR5noChemo': 'LR5(noChemoHistory)', 'LogisticRegression': 'LR16', 'XGBoost':'XGBoost',
                'NeuralNetwork2':'MultilayerPerceptron (2 layers)','LightGBM':'LightGBM', 'RandomForest':'RandomForest',
                'NeuralNetwork4':'MultilayerPerceptron (4 layers)', 'DNN':'KerasSequential',
                'NeuralNetwork3':'MultilayerPerceptron (3 layers)', 'NeuralNetwork1':'MultilayerPerceptron (1 layer)',
                'GaussianProcess':'GaussianProcess', 'GBoost':'GBoost', 'DecisionTree':'DecisionTree',
                'HGBoost':'HGBoost', 'RF6':'RandomForest6','AdaBoost':'AdaBoost', 'SupportVectorMachineRadial':'SupportVectorMachine',
                'RF16_NBT':'RandomForest16 (Chowell et al.)', 'kNearestNeighbourhood':'kNearestNeighbors', 'TMB':'TMB'}

try:
    randomSeed = int(sys.argv[1])
except:
    randomSeed = 1
CPU_num = -1
N_repeat_KFold_paramTune = 1
N_repeat_KFold = 2000
info_shown = 1
Kfold = 5 # 5
resampleNUM = N_repeat_KFold*Kfold

performance_df = pd.DataFrame()

MLM_performance_dict = {}

###### LLR6 model
MLM = 'LLR6'
temp_df = pd.DataFrame()
temp_df['method'] = [MLM] * resampleNUM
if MLM in MLM_list1:
    SCALE = 'None'
elif MLM in MLM_list2:
    SCALE = 'StandardScaler'
else:
    raise Exception('MLM not recognized!')
fnIn = '../03.Results/6features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(Kfold) + 'Rep' + str(
    N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
dataIn = open(fnIn, 'r').readlines()
header = dataIn[0].strip().split('\t')
position = header.index('params')
data = dataIn[1].strip().split('\t')
data = data[(position + 2):]
data = [float(c) for c in data]
temp_df['AUC_test'] = data[resampleNUM*0:resampleNUM*1]
temp_df['AUC_train'] = data[(resampleNUM * 1+3):(resampleNUM*2 + 3)]
temp_df['PRAUC_test'] = data[(resampleNUM*2 + 3+2):(resampleNUM*3 + 3+2)]
temp_df['PRAUC_train'] = data[(resampleNUM * 3 + 3*2 + 2):(resampleNUM * 4 + 3*2 + 2)]
temp_df['Accuracy_test'] = data[(resampleNUM * 4 + 3*2 + 2*2):(resampleNUM * 5 + 3*2 + 2*2)]
temp_df['Accuracy_train'] = data[(resampleNUM * 5 + 3*3 + 2*2):(resampleNUM * 6 + 3*3 + 2*2)]
temp_df['F1_test'] = data[(resampleNUM * 6 + 3*3 + 2*3):(resampleNUM * 7 + 3*3 + 2*3)]
temp_df['F1_train'] = data[(resampleNUM * 7 + 3*4 + 2*3):(resampleNUM * 8 + 3*4 + 2*3)]
temp_df['Performance_test'] = (temp_df['AUC_test']*temp_df['PRAUC_test']*temp_df['Accuracy_test']*temp_df['F1_test'])**(1/4)
temp_df['Performance_train'] = (temp_df['AUC_train']*temp_df['PRAUC_train']*temp_df['Accuracy_train']*temp_df['F1_train'])**(1/4)
temp_df['AUC_delta'] = temp_df['AUC_train'] - temp_df['AUC_test']
temp_df['PRAUC_delta'] = temp_df['PRAUC_train'] - temp_df['PRAUC_test']
temp_df['Accuracy_delta'] = temp_df['Accuracy_train'] - temp_df['Accuracy_test']
temp_df['F1_delta'] = temp_df['F1_train'] - temp_df['F1_test']
temp_df['Performance_delta'] = temp_df['Performance_train'] - temp_df['Performance_test']
performance_df = pd.concat([performance_df, temp_df], axis=0)
MLM_performance_dict[MLM] = [np.mean(temp_df['Performance_train']), np.mean(temp_df['Performance_test']), np.mean(temp_df['Performance_delta'])]


###### LR5 (without TMB) model
MLM = 'LR5noTMB'
temp_df = pd.DataFrame()
temp_df['method'] = [MLM] * resampleNUM
if MLM in MLM_list1:
    SCALE = 'None'
elif MLM in MLM_list2:
    SCALE = 'StandardScaler'
else:
    raise Exception('MLM not recognized!')
fnIn = '../03.Results/6features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(Kfold) + 'Rep' + str(
    N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
dataIn = open(fnIn, 'r').readlines()
header = dataIn[0].strip().split('\t')
position = header.index('params')
data = dataIn[1].strip().split('\t')
data = data[(position + 2):]
data = [float(c) for c in data]
temp_df['AUC_test'] = data[resampleNUM*0:resampleNUM*1]
temp_df['AUC_train'] = data[(resampleNUM * 1+3):(resampleNUM*2 + 3)]
temp_df['PRAUC_test'] = data[(resampleNUM*2 + 3+2):(resampleNUM*3 + 3+2)]
temp_df['PRAUC_train'] = data[(resampleNUM * 3 + 3*2 + 2):(resampleNUM * 4 + 3*2 + 2)]
temp_df['Accuracy_test'] = data[(resampleNUM * 4 + 3*2 + 2*2):(resampleNUM * 5 + 3*2 + 2*2)]
temp_df['Accuracy_train'] = data[(resampleNUM * 5 + 3*3 + 2*2):(resampleNUM * 6 + 3*3 + 2*2)]
temp_df['F1_test'] = data[(resampleNUM * 6 + 3*3 + 2*3):(resampleNUM * 7 + 3*3 + 2*3)]
temp_df['F1_train'] = data[(resampleNUM * 7 + 3*4 + 2*3):(resampleNUM * 8 + 3*4 + 2*3)]
temp_df['Performance_test'] = (temp_df['AUC_test']*temp_df['PRAUC_test']*temp_df['Accuracy_test']*temp_df['F1_test'])**(1/4)
temp_df['Performance_train'] = (temp_df['AUC_train']*temp_df['PRAUC_train']*temp_df['Accuracy_train']*temp_df['F1_train'])**(1/4)
temp_df['AUC_delta'] = temp_df['AUC_train'] - temp_df['AUC_test']
temp_df['PRAUC_delta'] = temp_df['PRAUC_train'] - temp_df['PRAUC_test']
temp_df['Accuracy_delta'] = temp_df['Accuracy_train'] - temp_df['Accuracy_test']
temp_df['F1_delta'] = temp_df['F1_train'] - temp_df['F1_test']
temp_df['Performance_delta'] = temp_df['Performance_train'] - temp_df['Performance_test']
performance_df = pd.concat([performance_df, temp_df], axis=0)
MLM_performance_dict[MLM] = [np.mean(temp_df['Performance_train']), np.mean(temp_df['Performance_test']), np.mean(temp_df['Performance_delta'])]

'''
###### LR5 (without Chemo history) model
MLM = 'LR5noChemo'
temp_df = pd.DataFrame()
temp_df['method'] = [MLM] * resampleNUM
if MLM in MLM_list1:
    SCALE = 'None'
elif MLM in MLM_list2:
    SCALE = 'StandardScaler'
else:
    raise Exception('MLM not recognized!')
fnIn = '../03.Results/6features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(Kfold) + 'Rep' + str(
    N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
dataIn = open(fnIn, 'r').readlines()
header = dataIn[0].strip().split('\t')
position = header.index('params')
data = dataIn[1].strip().split('\t')
data = data[(position + 2):]
data = [float(c) for c in data]
temp_df['AUC_test'] = data[resampleNUM*0:resampleNUM*1]
temp_df['AUC_train'] = data[(resampleNUM * 1+3):(resampleNUM*2 + 3)]
temp_df['PRAUC_test'] = data[(resampleNUM*2 + 3+2):(resampleNUM*3 + 3+2)]
temp_df['PRAUC_train'] = data[(resampleNUM * 3 + 3*2 + 2):(resampleNUM * 4 + 3*2 + 2)]
temp_df['Accuracy_test'] = data[(resampleNUM * 4 + 3*2 + 2*2):(resampleNUM * 5 + 3*2 + 2*2)]
temp_df['Accuracy_train'] = data[(resampleNUM * 5 + 3*3 + 2*2):(resampleNUM * 6 + 3*3 + 2*2)]
temp_df['F1_test'] = data[(resampleNUM * 6 + 3*3 + 2*3):(resampleNUM * 7 + 3*3 + 2*3)]
temp_df['F1_train'] = data[(resampleNUM * 7 + 3*4 + 2*3):(resampleNUM * 8 + 3*4 + 2*3)]
temp_df['Performance_test'] = (temp_df['AUC_test']*temp_df['PRAUC_test']*temp_df['Accuracy_test']*temp_df['F1_test'])**(1/4)
temp_df['Performance_train'] = (temp_df['AUC_train']*temp_df['PRAUC_train']*temp_df['Accuracy_train']*temp_df['F1_train'])**(1/4)
temp_df['AUC_delta'] = temp_df['AUC_train'] - temp_df['AUC_test']
temp_df['PRAUC_delta'] = temp_df['PRAUC_train'] - temp_df['PRAUC_test']
temp_df['Accuracy_delta'] = temp_df['Accuracy_train'] - temp_df['Accuracy_test']
temp_df['F1_delta'] = temp_df['F1_train'] - temp_df['F1_test']
temp_df['Performance_delta'] = temp_df['Performance_train'] - temp_df['Performance_test']
performance_df = pd.concat([performance_df, temp_df], axis=0)
MLM_performance_dict[MLM] = [np.mean(temp_df['Performance_train']), np.mean(temp_df['Performance_test']), np.mean(temp_df['Performance_delta'])]
'''

for MLM in MLM_list:
    print(MLM, ' in processing ...')

    temp_df = pd.DataFrame()
    temp_df['method'] = [MLM] * resampleNUM
    if MLM in MLM_list1:
        SCALE = 'None'
    elif MLM in MLM_list2:
        SCALE = 'StandardScaler'
    else:
        raise Exception('MLM not recognized!')
    fnIn = '../03.Results/16features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(Kfold) + 'Rep' + str(
        N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    dataIn = open(fnIn, 'r').readlines()
    header = dataIn[0].strip().split('\t')
    position = header.index('params')
    data = dataIn[1].strip().split('\t')
    data = data[(position+2):]
    data = [float(c) for c in data]
    temp_df['AUC_test'] = data[resampleNUM*0:resampleNUM*1]
    temp_df['AUC_train'] = data[(resampleNUM * 1+3):(resampleNUM*2 + 3)]
    temp_df['PRAUC_test'] = data[(resampleNUM*2 + 3+2):(resampleNUM*3 + 3+2)]
    temp_df['PRAUC_train'] = data[(resampleNUM * 3 + 3*2 + 2):(resampleNUM * 4 + 3*2 + 2)]
    temp_df['Accuracy_test'] = data[(resampleNUM * 4 + 3*2 + 2*2):(resampleNUM * 5 + 3*2 + 2*2)]
    temp_df['Accuracy_train'] = data[(resampleNUM * 5 + 3*3 + 2*2):(resampleNUM * 6 + 3*3 + 2*2)]
    temp_df['F1_test'] = data[(resampleNUM * 6 + 3*3 + 2*3):(resampleNUM * 7 + 3*3 + 2*3)]
    temp_df['F1_train'] = data[(resampleNUM * 7 + 3*4 + 2*3):(resampleNUM * 8 + 3*4 + 2*3)]
    temp_df['Performance_test'] = (temp_df['AUC_test'] * temp_df['PRAUC_test'] * temp_df['Accuracy_test'] * temp_df[
        'F1_test']) ** (1 / 4)
    temp_df['Performance_train'] = (temp_df['AUC_train'] * temp_df['PRAUC_train'] * temp_df['Accuracy_train'] * temp_df[
        'F1_train']) ** (1 / 4)
    temp_df['AUC_delta'] = temp_df['AUC_train'] - temp_df['AUC_test']
    temp_df['PRAUC_delta'] = temp_df['PRAUC_train'] - temp_df['PRAUC_test']
    temp_df['Accuracy_delta'] = temp_df['Accuracy_train'] - temp_df['Accuracy_test']
    temp_df['F1_delta'] = temp_df['F1_train'] - temp_df['F1_test']
    temp_df['Performance_delta'] = temp_df['Performance_train'] - temp_df['Performance_test']
    performance_df = pd.concat([performance_df, temp_df], axis=0)
    MLM_performance_dict[MLM] = [np.mean(temp_df['Performance_train']), np.mean(temp_df['Performance_test']), np.mean(temp_df['Performance_delta'])]


plot_type_list = ['train', 'test', 'delta']
plot_type_ind = 2
plot_train_test  = plot_type_list[plot_type_ind]  # train test delta
if plot_type_ind in [0,1]:
    MLM_order = [k for k, v in sorted(MLM_performance_dict.items(), key=lambda item: item[1][plot_type_ind], reverse=True)]
else:
    MLM_order = [k for k, v in sorted(MLM_performance_dict.items(), key=lambda item: item[1][plot_type_ind])]
cat_dtype = pd.api.types.CategoricalDtype(categories=MLM_order, ordered=True)
performance_df['method'] = performance_df['method'].astype(cat_dtype)
performance_df = performance_df.sort_values(by='method')

if plot_type_ind in [0,1]:
    model_train_performance = [v[0] for k, v in sorted(MLM_performance_dict.items(), key=lambda item: item[1][plot_type_ind], reverse=True)]
    model_test_performance = [v[1] for k, v in sorted(MLM_performance_dict.items(), key=lambda item: item[1][plot_type_ind], reverse=True)]
else:
    model_train_performance = [v[0] for k, v in sorted(MLM_performance_dict.items(), key=lambda item: item[1][plot_type_ind])]
    model_test_performance = [v[1] for k, v in sorted(MLM_performance_dict.items(), key=lambda item: item[1][plot_type_ind])]
model_delta_performance = [model_train_performance[i] - model_test_performance[i] for i in range(len(model_test_performance))]
for i in range(len(MLM_order)):
    print('%s  %.3f  %.3f'%(MLM_name_map[MLM_order[i]], model_test_performance[i], model_delta_performance[i]))
MLM_order = [MLM_name_map[MLM_order[i]] for i in range(len(MLM_order))]

boxplot_linewidth = 0.5
###################################### axes[0]: AUC_test ######################################
i=0
j=0
x_str = 'AUC_'+plot_train_test
if plot_train_test != 'delta':
    random_performance = 0.5
else:
    random_performance = 0
graph = sns.violinplot(y="method", x=x_str, data=performance_df,
                    palette=palette,linewidth=0,
                    scale="width", inner=None, ax = axes[j],zorder=2)
graph.axvline(random_performance, color='0.5', linestyle='--', linewidth=boxplot_linewidth)
for violin in axes[j].collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=axes[j].transData))

old_len_collections = len(axes[j].collections)
sns.stripplot(y="method", x=x_str, data=performance_df, color= '0.5', alpha=0.3, size=1, ax=axes[j],zorder=1) # dodgerblue edgecolor='black', linewidth=0.5,
for dots in axes[j].collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))

sns.boxplot(y="method", x=x_str, data=performance_df, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, color='k', linewidth=boxplot_linewidth, ax=axes[j]) # saturation=1,



###################################### axes[1]: PRAUC_test ######################################
i=0
j=1
x_str = 'PRAUC_'+plot_train_test
if plot_train_test != 'delta':
    random_performance = 409/1479
else:
    random_performance = 0
graph = sns.violinplot(y="method", x=x_str, data=performance_df,
                    palette=palette,linewidth=0,
                    scale="width", inner=None, ax = axes[j],zorder=2)
graph.axvline(random_performance, color='0.5', linestyle='--', linewidth=boxplot_linewidth)
for violin in axes[j].collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=axes[j].transData))

old_len_collections = len(axes[j].collections)
sns.stripplot(y="method", x=x_str, data=performance_df, color= '0.5', alpha=0.3, size=1, ax=axes[j],zorder=1) # dodgerblue edgecolor='black', linewidth=0.5,
for dots in axes[j].collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))

sns.boxplot(y="method", x=x_str, data=performance_df, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, color='k', linewidth=boxplot_linewidth, ax=axes[j]) # saturation=1,



###################################### axes[2]: Accuracy_test ######################################
i=0
j=2
x_str = 'Accuracy_'+plot_train_test
if plot_train_test != 'delta':
    random_performance = 0.5
else:
    random_performance = 0
graph = sns.violinplot(y="method", x=x_str, data=performance_df,
                    palette=palette,linewidth=0,
                    scale="width", inner=None, ax = axes[j],zorder=2)
graph.axvline(random_performance, color='0.5', linestyle='--', linewidth=boxplot_linewidth)
for violin in axes[j].collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=axes[j].transData))

old_len_collections = len(axes[j].collections)
sns.stripplot(y="method", x=x_str, data=performance_df, color= '0.5', alpha=0.3, size=1, ax=axes[j],zorder=1) # dodgerblue edgecolor='black', linewidth=0.5,
for dots in axes[j].collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))

sns.boxplot(y="method", x=x_str, data=performance_df, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, color='k', linewidth=boxplot_linewidth, ax=axes[j]) # saturation=1,



###################################### axes[3]: F1_test ######################################
i=0
j=3
x_str = 'F1_'+plot_train_test
if plot_train_test != 'delta':
    random_performance = 0.5
else:
    random_performance = 0
graph = sns.violinplot(y="method", x=x_str, data=performance_df,
                    palette=palette,linewidth=0,
                    scale="width", inner=None, ax = axes[j],zorder=2)
graph.axvline(random_performance, color='0.5', linestyle='--', linewidth=boxplot_linewidth)
for violin in axes[j].collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=axes[j].transData))

old_len_collections = len(axes[j].collections)
sns.stripplot(y="method", x=x_str, data=performance_df, color= '0.5', alpha=0.3, size=1, ax=axes[j],zorder=1) # dodgerblue edgecolor='black', linewidth=0.5,
for dots in axes[j].collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))

sns.boxplot(y="method", x=x_str, data=performance_df, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, color='k', linewidth=boxplot_linewidth, ax=axes[j]) # saturation=1,



###################################### axes[4]: Performance_test ######################################
i=0
j=4
x_str = 'Performance_'+plot_train_test
if plot_train_test != 'delta':
    random_performance = (0.5**3*(409/1479))**(1/4)
else:
    random_performance = 0
graph = sns.violinplot(y="method", x=x_str, data=performance_df,
                    palette=palette,linewidth=0,
                    scale="width", inner=None, ax = axes[j],zorder=2)
graph.axvline(random_performance, color='0.5', linestyle='--', linewidth=boxplot_linewidth)
for violin in axes[j].collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=axes[j].transData))

old_len_collections = len(axes[j].collections)
sns.stripplot(y="method", x=x_str, data=performance_df, color= '0.5', alpha=0.3, size=1, ax=axes[j],zorder=1) # dodgerblue edgecolor='black', linewidth=0.5,
for dots in axes[j].collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))

sns.boxplot(y="method", x=x_str, data=performance_df, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, color='k', linewidth=boxplot_linewidth, ax=axes[j]) # saturation=1,


for j in range(5):
    axes[j].spines.left.set_visible(False)
    axes[j].spines.right.set_visible(False)
    axes[j].spines.top.set_visible(False)
    axes[j].set_yticks([])
    axes[j].set_ylabel('')
    if plot_train_test != 'delta':
        axes[j].set_xlim([0, 1])
    else:
        axes[j].set_xlim([-0.2, 1])
if plot_train_test != 'delta':
    axes[0].set_xlabel('AUC')
    axes[1].set_xlabel('PRAUC')
    axes[2].set_xlabel('Accuracy')
    axes[3].set_xlabel('F1 score')
    axes[4].set_xlabel('Performance')
else:
    axes[0].set_xlabel('\u0394 AUC')
    axes[1].set_xlabel('\u0394 PRAUC')
    axes[2].set_xlabel('\u0394 Accuracy')
    axes[3].set_xlabel('\u0394 F1 score')
    axes[4].set_xlabel('\u0394 Performance')

axes[0].set_yticks(range(len(MLM_order)))
axes[0].set_yticklabels(MLM_order)
axes[0].tick_params(axis='y', direction='out', length=0, width=0, pad=0)

fnOut = '../03.Results/FigureS4_PanCancer_ModelPerformance_'+plot_train_test+'.png'
plt.savefig(fnOut,dpi=1200)
