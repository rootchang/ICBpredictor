import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Arial"
palette = sns.color_palette("deep")

def AUC_calculator(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    return auroc, threshold[ind_max]

def AUPRC_calculator(y, y_pred):
    prec, recall, threshold = precision_recall_curve(y, y_pred)
    AUPRC = auc(recall, prec)
    specificity_sensitivity_sum = recall + (1 - prec)
    ind_max = np.argmax(specificity_sensitivity_sum)
    return AUPRC, threshold[ind_max]

if __name__ == "__main__":
    start_time = time.time()

    CPU_num = -1
    randomSeed = 1

    fix_cutoff = 1
    cutoff_value_LLR6 = 0.44
    cutoff_value_PDL1 = 50
    cutoff_value_TMB = 10

    ########################## Read in data ##########################
    phenoNA = 'Response'
    LLRmodelNA = sys.argv[1]  # 'LLR6'   'LLR5noTMB'   'LLR5noChemo'
    if LLRmodelNA == 'LLR6':
        cutoff_value_LLR6 = 0.44
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
    elif LLRmodelNA == 'LLR5noTMB':
        cutoff_value_LLR6 = 0.48
        featuresNA = ['PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
    elif LLRmodelNA == 'LLR5noChemo':
        cutoff_value_LLR6 = 0.43
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Albumin', 'NLR', 'Age']
    xy_colNAs = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age'] + [phenoNA]

    print('Raw data processing ...')
    dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
    dataChowellTrain = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
    dataChowellTest = pd.read_excel(dataALL_fn, sheet_name='Chowell2018', index_col=0)
    dataChowell = pd.concat([dataChowellTrain, dataChowellTest], axis=0)
    dataChowell_NSCLC = dataChowell.loc[dataChowell['CancerType'] == 'NSCLC',]
    dataChowell_Gastric = dataChowell.loc[dataChowell['CancerType'] == 'Gastric',]
    dataChowell_Mesothelioma = dataChowell.loc[dataChowell['CancerType'] == 'Mesothelioma',]
    dataChowell_Esophageal = dataChowell.loc[dataChowell['CancerType'] == 'Esophageal',]

    dataMorris_new = pd.read_excel(dataALL_fn, sheet_name='Morris_new', index_col=0)
    dataMorris_new2 = pd.read_excel(dataALL_fn, sheet_name='Morris_new2', index_col=0)
    dataMorris = pd.concat([dataMorris_new, dataMorris_new2], axis=0)
    dataMorris_Gastric = dataMorris.loc[dataMorris['CancerType'] == 'Gastric',]
    dataMorris_Mesothelioma = dataMorris.loc[dataMorris['CancerType'] == 'Mesothelioma',]
    dataMorris_Esophageal = dataMorris.loc[dataMorris['CancerType'] == 'Esophageal',]

    dataMSK_Gastric = pd.concat([dataChowell_Gastric[xy_colNAs], dataMorris_Gastric[xy_colNAs]], axis=0)
    dataMSK_Mesothelioma = pd.concat([dataChowell_Mesothelioma[xy_colNAs], dataMorris_Mesothelioma[xy_colNAs]], axis=0)
    dataMSK_Esophageal = pd.concat([dataChowell_Esophageal[xy_colNAs], dataMorris_Esophageal[xy_colNAs]], axis=0)

    dataAwad1 = pd.read_excel(dataALL_fn, sheet_name='Awad_NSCLC1', index_col=0)
    dataAwad2 = pd.read_excel(dataALL_fn, sheet_name='Awad_NSCLC2', index_col=0)
    dataAwad3 = pd.read_excel(dataALL_fn, sheet_name='Awad_NSCLC3', index_col=0)
    dataAwad = pd.concat([dataAwad1, dataAwad2,dataAwad3], axis=0)
    dataLee = pd.read_excel(dataALL_fn, sheet_name='Lee_NSCLC', index_col=0)

    dataALL = [dataChowell_NSCLC, dataMSK_Gastric, dataMSK_Mesothelioma, dataMSK_Esophageal]
    for i in range(len(dataALL)):
        dataALL[i] = dataALL[i][xy_colNAs].astype(float)
        dataALL[i] = dataALL[i].dropna(axis=0)

    # truncate TMB
    TMB_upper = 50
    try:
        for i in range(len(dataALL)):
            dataALL[i]['TMB'] = [c if c<TMB_upper else TMB_upper for c in dataALL[i]['TMB']]
    except:
        1
    # truncate Age
    Age_upper = 85
    try:
        for i in range(len(dataALL)):
            dataALL[i]['Age'] = [c if c < Age_upper else Age_upper for c in dataALL[i]['Age']]
    except:
        1
    # truncate NLR
    NLR_upper = 25
    try:
        for i in range(len(dataALL)):
            dataALL[i]['NLR'] = [c if c < NLR_upper else NLR_upper for c in dataALL[i]['NLR']]
    except:
        1

    x_test_list = []
    y_test_list = []
    for c in dataALL:
        x_test_list.append(pd.DataFrame(c, columns=xy_colNAs))
        y_test_list.append(c[phenoNA])

    y_pred_LLR6 = []
    Sensitivity_LLR6 = []
    Specificity_LLR6 = []
    Accuracy_LLR6 = []
    PPV_LLR6 = []
    NPV_LLR6 = []
    F1_LLR6 = []
    OddsRatio_LLR6 = []

    y_pred_PDL1 = []
    Sensitivity_PDL1 = []
    Specificity_PDL1 = []
    Accuracy_PDL1 = []
    PPV_PDL1 = []
    NPV_PDL1 = []
    F1_PDL1 = []
    OddsRatio_PDL1 = []

    y_pred_TMB = []
    Sensitivity_TMB = []
    Specificity_TMB = []
    Accuracy_TMB = []
    PPV_TMB = []
    NPV_TMB = []
    F1_TMB = []
    OddsRatio_TMB = []

    ###################### Read in LLR model params ######################
    fnIn = '../03.Results/16features/NSCLC/NSCLC_'+LLRmodelNA+'_10k_ParamCalculate.txt'
    params_data = open(fnIn,'r').readlines()
    params_dict = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict[param_name] = params_val

    ########################## test LLR6 model performance ##########################
    x_test_scaled_list = []
    scaler_sd = StandardScaler()
    scaler_sd.fit(x_test_list[0][featuresNA])
    scaler_sd.mean_ = np.array(params_dict['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict['LLR_scale'])
    for c in x_test_list:
        x_test_scaled_list.append(pd.DataFrame(scaler_sd.transform(c[featuresNA])))
    clf = linear_model.LogisticRegression().fit(x_test_scaled_list[0], y_test_list[0])
    clf.coef_ = np.array([params_dict['LLR_coef']])
    clf.intercept_ = np.array(params_dict['LLR_intercept'])

    print('LLR6_meanParams10000:')
    fnOut = '../03.Results/NSCLC_' + LLRmodelNA + '_Scaler(' + 'StandardScaler' + ')_prediction.xlsx'
    dataALL[0].to_excel(fnOut, sheet_name='0')
    for i in range(len(x_test_scaled_list)):
        y_pred_test = clf.predict_proba(x_test_scaled_list[i])[:, 1]
        dataALL[i][LLRmodelNA] = y_pred_test
        dataALL[i].to_csv('../03.Results/NSCLCmodel_'+LLRmodelNA+'_Dataset'+str(i+1)+'.csv', index=True) # PanCancermodel_Dataset   NSCLCmodel_Dataset
        AUC_test, score_test = AUC_calculator(y_test_list[i], y_pred_test)
        print('   Dataset %d: %5.3f (n=%d) %8.3f' % (i+1, AUC_test, len(y_pred_test), score_test))

        content = dataALL[i].loc[:,['Response',LLRmodelNA]]
        content.rename(columns=dict(zip(content.columns, ['y','y_pred'])), inplace=True)
        with pd.ExcelWriter(fnOut, engine="openpyxl", mode='a',if_sheet_exists="replace") as writer:
            content.to_excel(writer, sheet_name=str(i))

        if fix_cutoff:
            score = cutoff_value_LLR6
        else:
            AUC, score = AUC_calculator(y_test_list[i], y_pred_test)
        y_pred_01 = [int(c >= score) for c in y_pred_test]
        tn, fp, fn, tp = confusion_matrix(y_test_list[i], y_pred_01).ravel()
        Sensitivity = tp / (tp + fn)  # TPR, recall
        Specificity = tn / (tn + fp)  # 1 - FPR
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        PPV = tp / (tp + fp)  # Precision
        NPV = tn / (tn + fn)
        F1 = 2*PPV*Sensitivity/(PPV+Sensitivity)
        OddsRatio = (tp / (tp + fp)) / (fn / (tn + fn))

        y_pred_LLR6.append(y_pred_test)
        Sensitivity_LLR6.append(Sensitivity)
        Specificity_LLR6.append(Specificity)
        Accuracy_LLR6.append(Accuracy)
        PPV_LLR6.append(PPV)
        NPV_LLR6.append(NPV)
        F1_LLR6.append(F1)
        OddsRatio_LLR6.append(OddsRatio)

    ########################## test PDL1 model performance ##########################
    modelNA = 'PDL1'

    print('PDL1:')
    fnOut = '../03.Results/NSCLC_' + modelNA + '_Scaler(' + 'None' + ')_prediction.xlsx'
    dataALL[0].to_excel(fnOut, sheet_name='0')
    for i in range(len(x_test_list)):
        y_pred_test = x_test_list[i]['PDL1_TPS(%)']
        dataALL[i]['PDL1_TPS(%)'] = y_pred_test
        dataALL[i].to_csv('../03.Results/NSCLCmodel_'+modelNA+'_Dataset'+str(i+1)+'.csv', index=True) # PanCancermodel_Dataset   NSCLCmodel_Dataset
        AUC_test, score_test = AUC_calculator(y_test_list[i], y_pred_test)
        print('   Dataset %d: %5.3f (n=%d) %8.3f' % (i+1, AUC_test, len(y_pred_test), score_test))

        content = dataALL[i].loc[:,['Response','PDL1_TPS(%)']]
        content.rename(columns=dict(zip(content.columns, ['y','y_pred'])), inplace=True)
        with pd.ExcelWriter(fnOut, engine="openpyxl", mode='a',if_sheet_exists="replace") as writer:
            content.to_excel(writer, sheet_name=str(i))

        if fix_cutoff:
            score = cutoff_value_PDL1
        else:
            AUC, score = AUC_calculator(y_test_list[i], y_pred_test)
        y_pred_01 = [int(c >= score) for c in y_pred_test]
        tn, fp, fn, tp = confusion_matrix(y_test_list[i], y_pred_01).ravel()
        Sensitivity = tp / (tp + fn)  # TPR, recall
        Specificity = tn / (tn + fp)  # 1 - FPR
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        PPV = tp / (tp + fp)  # Precision
        NPV = tn / (tn + fn)
        F1 = 2 * PPV * Sensitivity / (PPV + Sensitivity)
        OddsRatio = (tp / (tp + fp)) / (fn / (tn + fn))

        y_pred_PDL1.append(y_pred_test)
        Sensitivity_PDL1.append(Sensitivity)
        Specificity_PDL1.append(Specificity)
        Accuracy_PDL1.append(Accuracy)
        PPV_PDL1.append(PPV)
        NPV_PDL1.append(NPV)
        F1_PDL1.append(F1)
        OddsRatio_PDL1.append(OddsRatio)

    ########################## test TMB model performance ##########################
    modelNA = 'TMB' # TMB NLR Albumin Age

    print(modelNA+':')
    fnOut = '../03.Results/NSCLC_' + modelNA + '_Scaler(' + 'None' + ')_prediction.xlsx'
    dataALL[0].to_excel(fnOut, sheet_name='0')
    for i in range(len(x_test_list)):
        y_pred_test = x_test_list[i][modelNA]
        dataALL[i][modelNA] = y_pred_test
        dataALL[i].to_csv('../03.Results/NSCLCmodel_'+modelNA+'_Dataset'+str(i+1)+'.csv', index=True) # PanCancermodel_Dataset   NSCLCmodel_Dataset
        AUC_test, score_test = AUC_calculator(y_test_list[i], y_pred_test)
        print('   Dataset %d: %5.3f (n=%d) %8.3f' % (i+1, AUC_test, len(y_pred_test), score_test))

        content = dataALL[i].loc[:,['Response',modelNA]]
        content.rename(columns=dict(zip(content.columns, ['y','y_pred'])), inplace=True)
        with pd.ExcelWriter(fnOut, engine="openpyxl", mode='a',if_sheet_exists="replace") as writer:
            content.to_excel(writer, sheet_name=str(i))

        if not fix_cutoff:
            AUC, score = AUC_calculator(y_test_list[i], y_pred_test)
        else:
            score = cutoff_value_TMB
        y_pred_01 = [int(c >= score) for c in y_pred_test]
        tn, fp, fn, tp = confusion_matrix(y_test_list[i], y_pred_01).ravel()
        Sensitivity = tp / (tp + fn)  # TPR, recall
        Specificity = tn / (tn + fp)  # 1 - FPR
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        PPV = tp / (tp + fp)  # Precision
        NPV = tn / (tn + fn)
        F1 = 2 * PPV * Sensitivity / (PPV + Sensitivity)
        OddsRatio = (tp / (tp + fp)) / (fn / (tn + fn))

        y_pred_TMB.append(y_pred_test)
        Sensitivity_TMB.append(Sensitivity)
        Specificity_TMB.append(Specificity)
        Accuracy_TMB.append(Accuracy)
        PPV_TMB.append(PPV)
        NPV_TMB.append(NPV)
        F1_TMB.append(F1)
        OddsRatio_TMB.append(OddsRatio)

    ############################## Plot ##############################
    textSize = 8

    ############# Plot ROC curves ##############
    output_fig1 = '../03.Results/NSCLC_'+LLRmodelNA+'_PDL1_TMB_ROC_compare_on_nonNSCLC.pdf'
    ax1 = [0] * 4
    fig1, ((ax1[0], ax1[1], ax1[2], ax1[3])) = plt.subplots(1, 4, figsize=(6.5, 1.5))
    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.96, wspace=0.3, hspace=0.35)

    for i in range(4):
        y_true = y_test_list[i]
        ###### LLR6 model
        y_pred = y_pred_LLR6[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        specificity_sensitivity_sum = tpr + (1 - fpr)
        ind_max = np.argmax(specificity_sensitivity_sum)
        if ind_max < 0.5:  # the first threshold is larger than all x values (tpr=1, fpr=1)
            ind_max = 1
        opt_cutoff = thresholds[ind_max]
        AUC = auc(fpr, tpr)
        ax1[i].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
        ax1[i].plot(fpr, tpr, color= palette[0],linestyle='-', label='%s (AUC: %.2f)' % (LLRmodelNA[0:4],AUC))
        ###### PDL1 model
        y_pred = y_pred_PDL1[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC = auc(fpr, tpr)
        ax1[i].plot(fpr, tpr, color= palette[2],linestyle='-', label='PDL1 (AUC: %.2f)' % (AUC))
        ###### TMB model
        y_pred = y_pred_TMB[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC = auc(fpr, tpr)
        ax1[i].plot(fpr, tpr, color= palette[3],linestyle='-', label='TMB (AUC: %.2f)' % (AUC))

        ax1[i].legend(frameon=False, loc=(0.2,-0.04), prop={'size': textSize},handlelength=1,handletextpad=0.1,
                      labelspacing = 0.2)
        ax1[i].set_xlim([-0.02, 1.02])
        ax1[i].set_ylim([-0.02, 1.02])
        ax1[i].set_yticks([0,0.5,1])
        ax1[i].set_xticks([0,0.5,1])
        if i > 0:
            ax1[i].set_yticklabels([])
        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)

    fig1.savefig(output_fig1) # , dpi=300
    plt.close()

    ############# Plot PRC curves ##############
    output_fig1 = '../03.Results/NSCLC_'+LLRmodelNA+'_PDL1_TMB_PRC_compare_on_nonNSCLC.pdf'
    ax1 = [0] * 4
    fig1, ((ax1[0], ax1[1], ax1[2], ax1[3])) = plt.subplots(1, 4, figsize=(6.5, 1.5))
    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.96, wspace=0.3, hspace=0.35)

    for i in range(4):
        y_true = y_test_list[i]
        ###### LLR6 model
        y_pred = y_pred_LLR6[i]
        prec, recall, _ = precision_recall_curve(y_true, y_pred) # , pos_label=clf.classes_[1]
        AUPRC, _ = AUPRC_calculator(y_true, y_pred)
        ax1[i].plot([0, 1], [sum(y_true)/len(y_true), sum(y_true)/len(y_true)], 'k', alpha=0.5, linestyle='--')
        ax1[i].plot(recall, prec, color= palette[0],linestyle='-', label='LLR6 (AUC: %.2f)' % (AUPRC))
        ###### PDL1 model
        y_pred = y_pred_PDL1[i]
        prec, recall, _ = precision_recall_curve(y_true, y_pred)  # , pos_label=clf.classes_[1]
        AUPRC, _ = AUPRC_calculator(y_true, y_pred)
        ax1[i].plot(recall, prec, color= palette[2],linestyle='-', label='PDL1 (AUC: %.2f)' % (AUPRC))
        ###### TMB model
        y_pred = y_pred_TMB[i]
        prec, recall, _ = precision_recall_curve(y_true, y_pred)  # , pos_label=clf.classes_[1]
        AUPRC, _ = AUPRC_calculator(y_true, y_pred)
        ax1[i].plot(recall, prec, color= palette[3],linestyle='-', label='TMB (AUC: %.2f)' % (AUPRC))

        ax1[i].legend(frameon=False, loc=(0.25, 0.7), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                      labelspacing=0.2)
        ax1[i].set_xlim([-0.02, 1.02])
        ax1[i].set_ylim([-0.02, 1.02])
        ax1[i].set_yticks([0, 0.5, 1])
        ax1[i].set_xticks([0, 0.5, 1])
        if i > 0:
            ax1[i].set_yticklabels([])
        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)

    fig1.savefig(output_fig1) # , dpi=300
    plt.close()

    ############# Plot confusion matrix ##############
    diplay_ratio = 1
    output_fig1 = '../03.Results/NSCLC_'+LLRmodelNA+'_PDL1_TMB_CM_compare_on_nonNSCLC.pdf'
    fig1, axes = plt.subplots(3, 4, figsize=(6.5, 6.5))
    fig1.subplots_adjust(left=0.05, bottom=0.05, right=0.97, top=0.96, wspace=0.4, hspace=0.5)

    for i in range(4):
        y_true = y_test_list[i]
        ###### LLR6 model
        y_pred = y_pred_LLR6[i]
        if fix_cutoff:
            score = cutoff_value_LLR6
        else:
            AUC, score = AUC_calculator(y_true, y_pred)
        y_pred_01 = [int(c >= score) for c in y_pred]
        cf_matrix = confusion_matrix(y_true, y_pred_01)
        if diplay_ratio:
            cf_matrix = np.array([[cf_matrix[0,0]/sum(cf_matrix[0]),cf_matrix[0,1]/sum(cf_matrix[0])], [cf_matrix[1,0]/sum(cf_matrix[1]),cf_matrix[1,1]/sum(cf_matrix[1])]])
        disp = ConfusionMatrixDisplay(cf_matrix, display_labels=['NR', 'R'])
        disp.plot(ax=axes[0][i],cmap=plt.cm.Blues) # , xticks_rotation=45
        disp.im_.set_clim(0, 1)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
        ###### PDL1 model
        y_pred = y_pred_PDL1[i]
        if fix_cutoff:
            score = cutoff_value_PDL1
        else:
            AUC, score = AUC_calculator(y_true, y_pred)
        y_pred_01 = [int(c >= score) for c in y_pred]
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred_01, normalize='true', include_values = False, display_labels=['NR', 'R'])
        disp.plot(ax=axes[1][i],cmap=plt.cm.Blues)  # , xticks_rotation=45
        disp.im_.set_clim(0, 1)
        if i<30:
            disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
        ###### TMB model
        y_pred = y_pred_TMB[i]
        AUC, score = AUC_calculator(y_true, y_pred)
        if not fix_cutoff:
            AUC, score = AUC_calculator(y_true, y_pred)
        else:
            score = cutoff_value_TMB
        y_pred_01 = [int(c >= score) for c in y_pred]
        cf_matrix = confusion_matrix(y_true, y_pred_01)
        if diplay_ratio:
            cf_matrix = np.array([[cf_matrix[0,0]/sum(cf_matrix[0]),cf_matrix[0,1]/sum(cf_matrix[0])], [cf_matrix[1,0]/sum(cf_matrix[1]),cf_matrix[1,1]/sum(cf_matrix[1])]])
        disp = ConfusionMatrixDisplay(cf_matrix, display_labels=['NR', 'R'])
        disp.plot(ax=axes[2][i],cmap=plt.cm.Blues)  # , xticks_rotation=45
        disp.im_.set_clim(0, 1)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')

    fig1.savefig(output_fig1) # , dpi=300
    plt.close()

    ############# Plot metrics barplot ##############
    output_fig_fn = '../03.Results/NSCLC_'+LLRmodelNA+'_MultipleMetricComparison_on_nonNSCLC.pdf'
    plt.figure(figsize=(6.5, 4.5))
    ax1 = [0] * 6
    fig1, ((ax1[0], ax1[1], ax1[2]), (ax1[3], ax1[4], ax1[5])) = plt.subplots(2, 3, figsize=(6.5, 4))
    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.96, wspace=0.45, hspace=0.55)
    barWidth = 0.2
    color_list = [palette[0], palette[2], palette[3]]
    modelNA_list = ['LLR6', 'PDL1', 'TMB']
    metricsNA_list = ['Accuracy', 'F1 score','Odds ratio','Specificity','PPV','NPV']  # 'Sensitivity'
    LLR6_data = [Accuracy_LLR6, F1_LLR6, OddsRatio_LLR6, Specificity_LLR6, PPV_LLR6, NPV_LLR6] # Sensitivity_LLR6
    PDL1_data = [Accuracy_PDL1, F1_PDL1, OddsRatio_PDL1, Specificity_PDL1, PPV_PDL1, NPV_PDL1]
    TMB_data = [Accuracy_TMB, F1_TMB, OddsRatio_TMB, Specificity_TMB, PPV_TMB, NPV_TMB]

    #### with training set
    for i in range(6):
        bh11 = ax1[i].bar(np.array([0, 1, 2, 3]) + barWidth * 1, LLR6_data[i],
                       color=color_list[0], width=barWidth, edgecolor='k', label=modelNA_list[0])
        bh12 = ax1[i].bar(np.array([0, 1, 2, 3]) + barWidth * 2, PDL1_data[i],
                       color=color_list[1], width=barWidth, edgecolor='k', label=modelNA_list[1])
        bh13 = ax1[i].bar(np.array([0, 1, 2, 3]) + barWidth * 3, TMB_data[i],
                       color=color_list[2], width=barWidth, edgecolor='k', label=modelNA_list[2])

        ax1[i].set_xticks(np.array([0, 1, 2, 3]) + barWidth * 2)
        ax1[i].set_xticklabels([])
        if i == 0:
            ax1[i].legend(frameon=False, loc=(0.03, 0.9), prop={'size': textSize}, handlelength=1, ncol=3,
                      handletextpad=0.1, labelspacing=0.2)
        ax1[i].set_xlim([0, 4])
        if i == 2:
            ax1[i].set_ylim([0, 5])
            ax1[i].set_yticks([0,1,2,3,4])
        else:
            ax1[i].set_ylim([0, 1])
            ax1[i].set_yticks([0, 0.5, 1])
        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)
        ax1[i].set_ylabel(metricsNA_list[i])
        ax1[i].tick_params('x', length=0, width=0, which='major')

    fig1.savefig(output_fig_fn) # , dpi=300
    plt.close()