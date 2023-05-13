import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

if __name__ == "__main__":
    start_time = time.time()

    dataType = sys.argv[1] # PanCancer  NSCLC
    LLRmodelNA = sys.argv[2]  # 'LLR6'   'LLR5noTMB'   'LLR5noChemo'
    if dataType == 'NSCLC':
        if LLRmodelNA == 'LLR6':
            LLR_cutoff = 0.44
        elif LLRmodelNA == 'LLR5noTMB':
            LLR_cutoff = 0.48
        elif LLRmodelNA == 'LLR5noChemo':
            LLR_cutoff = 0.43
    else:
        if LLRmodelNA == 'LLR6':
            LLR_cutoff = 0.50
        elif LLRmodelNA == 'LLR5noTMB':
            LLR_cutoff = 0.53
        elif LLRmodelNA == 'LLR5noChemo':
            LLR_cutoff = 0.51
    TMB_cutoff = 10
    PDL1_cutoff = 50

    print('Raw data read in ...')
    fnIn = '../03.Results/' + dataType + '_' + LLRmodelNA + '_Scaler(StandardScaler)_prediction.xlsx'
    y_pred_LLR6 = []
    y_true = []
    start_set = 1
    if dataType == 'NSCLC':
        end_set = 4
    else:
        end_set = 3
    ##################################### LLR vs TMB  #####################################
    if start_set:
        output_fig_fn = '../03.Results/' + dataType + '_' + LLRmodelNA + '_TMB_scatterPlot_testOnly.pdf'
    else:
        output_fig_fn = '../03.Results/' + dataType + '_' + LLRmodelNA + '_TMB_scatterPlot_all.pdf'
    for sheet_i in range(start_set,end_set):
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_LLR6.extend(data['y_pred'].tolist())
        y_true.extend(data['y'].tolist())

    fnIn = '../03.Results/' + dataType + '_' + 'TMB_Scaler(None)_prediction.xlsx'
    y_pred_TMB = []
    for sheet_i in range(start_set,end_set):
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_TMB.extend(data['y_pred'].tolist())

    print('*************************Total (test) patient number: ', len(y_pred_TMB))

    log_shift = 4
    y_true = np.array(y_true)
    y_pred_LLR6 = np.array(y_pred_LLR6)
    y_pred_LLR6_R = y_pred_LLR6[y_true == 1]
    y_pred_LLR6_NR = y_pred_LLR6[y_true == 0]
    ORR_lowLLR6 = y_true[y_pred_LLR6 < LLR_cutoff]
    print('ORR low',LLRmodelNA,': ', sum(ORR_lowLLR6) / len(ORR_lowLLR6))
    ORR_highLLR6 = y_true[y_pred_LLR6 >= LLR_cutoff]
    print('ORR high',LLRmodelNA,': ', sum(ORR_highLLR6) / len(ORR_highLLR6))

    y_pred_TMB = np.array(y_pred_TMB)
    y_pred_TMB_R = np.log2(y_pred_TMB[y_true == 1]+log_shift)
    y_pred_TMB_NR = np.log2(y_pred_TMB[y_true == 0]+log_shift)
    ORR_lowTMB = y_true[y_pred_TMB < TMB_cutoff]
    LLRscore_lowTMB = y_pred_LLR6[y_pred_TMB < TMB_cutoff]
    ORR_R1 = ORR_lowTMB[LLRscore_lowTMB >= LLR_cutoff]
    ORR_R1 = sum(ORR_R1)/len(ORR_R1)
    ORR_R2 = ORR_lowTMB[LLRscore_lowTMB < LLR_cutoff]
    ORR_R2 = sum(ORR_R2) / len(ORR_R2)
    print('ORR R1: ', ORR_R1) # lowTMB-highLORIS
    print('ORR R2: ', ORR_R2) # lowTMB-lowLORIS
    print('ORR lowTMB: ', sum(ORR_lowTMB)/len(ORR_lowTMB))

    ORR_highTMB = y_true[y_pred_TMB >= TMB_cutoff]
    LLRscore_highTMB = y_pred_LLR6[y_pred_TMB >= TMB_cutoff]
    ORR_R3 = ORR_highTMB[LLRscore_highTMB >= LLR_cutoff]
    ORR_R3 = sum(ORR_R3) / len(ORR_R3)
    ORR_R4 = ORR_highTMB[LLRscore_highTMB < LLR_cutoff]
    ORR_R4 = sum(ORR_R4) / len(ORR_R4)
    print('ORR R3: ', ORR_R3)
    print('ORR R4: ', ORR_R4)
    print('ORR highTMB: ', sum(ORR_highTMB) / len(ORR_highTMB))

    ################ plot
    fontSize = 10
    plt.rcParams['font.size'] = fontSize
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(1, 1, figsize=(2.15*1.4, 1.9*1.4))
    plt.subplots_adjust(left=0.23, bottom=0.27, right=0.97, top=0.95, wspace=0.4, hspace=0.45)
    plt.scatter(y_pred_TMB_NR, y_pred_LLR6_NR, s=15, c='k', marker='^', alpha=1, linewidths=0)
    plt.scatter(y_pred_TMB_R, y_pred_LLR6_R, s=10, c='g', marker='o', alpha=1, linewidths=0)

    axes.set_ylim([-0.05,1])
    axes.set_yticks([0, 0.5, 1])
    axes.set_xlim([np.log2(log_shift)-0.1, 5.8])
    axes.set_xticks([np.log2(log_shift), np.log2(5+log_shift), np.log2(10+log_shift), np.log2(25+log_shift), np.log2(50+log_shift)])
    axes.set_xticklabels([0,5,10,25,50])
    axes.set_xlabel('TMB')
    axes.set_ylabel('LORIS')
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    axes.axvspan(np.log2(log_shift)-0.1, np.log2(10+log_shift), ymin=0, ymax=(1+0.05)*LLR_cutoff, facecolor='0.5', alpha=0.3, zorder=0)
    axes.axvspan(np.log2(log_shift)-0.1, np.log2(10+log_shift), ymin=(1+0.05)*LLR_cutoff, ymax=1, facecolor='g', alpha=0.1, zorder=0)
    axes.axvspan(np.log2(10 + log_shift),np.log2(50 + log_shift), ymin=0, ymax=(1+0.05)*LLR_cutoff, facecolor='0.5', alpha=0.1,zorder=0)
    axes.axvspan(np.log2(10 + log_shift),np.log2(50 + log_shift), ymin=(1+0.05)*LLR_cutoff, ymax=1, facecolor='g', alpha=0.3,zorder=0)

    fig.savefig(output_fig_fn) # , dpi=300
    plt.close()





    ##################################### LLR vs PD-L1  #####################################
    if dataType == 'NSCLC':
        if start_set:
            output_fig_fn = '../03.Results/' + dataType + '_' + LLRmodelNA + '_PDL1_scatterPlot_testOnly.pdf'
        else:
            output_fig_fn = '../03.Results/' + dataType + '_' + LLRmodelNA + '_PDL1_scatterPlot_all.pdf'

        fnIn = '../03.Results/' + dataType + '_' + 'PDL1_Scaler(None)_prediction.xlsx'
        y_pred_PDL1 = []
        for sheet_i in range(start_set,end_set):
            data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
            y_pred_PDL1.extend(data['y_pred'].tolist())

        print('*************************Total (test) patient number: ', len(y_pred_PDL1))

        y_true = np.array(y_true)
        y_pred_LLR6 = np.array(y_pred_LLR6)
        y_pred_LLR6_R = y_pred_LLR6[y_true == 1]
        y_pred_LLR6_NR = y_pred_LLR6[y_true == 0]
        ORR_lowLLR6 = y_true[y_pred_LLR6 < LLR_cutoff]
        print('ORR low',LLRmodelNA,': ', sum(ORR_lowLLR6) / len(ORR_lowLLR6))
        ORR_highLLR6 = y_true[y_pred_LLR6 >= LLR_cutoff]
        print('ORR high',LLRmodelNA,': ', sum(ORR_highLLR6) / len(ORR_highLLR6))

        y_pred_PDL1 = np.array(y_pred_PDL1)
        y_pred_PDL1_R = y_pred_PDL1[y_true == 1]
        y_pred_PDL1_NR = y_pred_PDL1[y_true == 0]
        ORR_lowPDL1 = y_true[y_pred_PDL1 < PDL1_cutoff]
        LLRscore_lowPDL1 = y_pred_LLR6[y_pred_PDL1 < PDL1_cutoff]
        ORR_R1 = ORR_lowPDL1[LLRscore_lowPDL1 >= LLR_cutoff]
        ORR_R1 = sum(ORR_R1)/len(ORR_R1)
        ORR_R2 = ORR_lowPDL1[LLRscore_lowPDL1 < LLR_cutoff]
        ORR_R2 = sum(ORR_R2) / len(ORR_R2)
        print('ORR R1: ', ORR_R1) # lowPDL1-highLORIS
        print('ORR R2: ', ORR_R2) # lowPDL1-lowLORIS
        print('ORR lowPDL1: ', sum(ORR_lowPDL1)/len(ORR_lowPDL1))

        ORR_highPDL1 = y_true[y_pred_PDL1 >= PDL1_cutoff]
        LLRscore_highPDL1 = y_pred_LLR6[y_pred_PDL1 >= PDL1_cutoff]
        ORR_R3 = ORR_highPDL1[LLRscore_highPDL1 >= LLR_cutoff]
        ORR_R3 = sum(ORR_R3) / len(ORR_R3)
        ORR_R4 = ORR_highPDL1[LLRscore_highPDL1 < LLR_cutoff]
        ORR_R4 = sum(ORR_R4) / len(ORR_R4)
        print('ORR R3: ', ORR_R3)
        print('ORR R4: ', ORR_R4)
        print('ORR highPDL1: ', sum(ORR_highPDL1) / len(ORR_highPDL1))

        ################ plot
        fontSize = 10
        plt.rcParams['font.size'] = fontSize
        plt.rcParams["font.family"] = "Arial"
        fig, axes = plt.subplots(1, 1, figsize=(2.15*1.4, 1.9*1.4))
        plt.subplots_adjust(left=0.23, bottom=0.27, right=0.97, top=0.95, wspace=0.4, hspace=0.45)
        plt.scatter(y_pred_PDL1_NR, y_pred_LLR6_NR, s=15, c='k', marker='^', alpha=1, linewidths=0)
        plt.scatter(y_pred_PDL1_R, y_pred_LLR6_R, s=10, c='g', marker='o', alpha=1, linewidths=0)

        axes.set_ylim([-0.05,1])
        axes.set_yticks([0, 0.5, 1])
        axes.set_xlim([0, 102])
        axes.set_xticks([0, 25, 50, 75, 100])
        axes.set_xlabel('PD-L1 TPS (%)')
        axes.set_ylabel('LORIS')
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.axvspan(0, 50, ymin=0, ymax=LLR_cutoff * 105 / 100, facecolor='0.5', alpha=0.3,zorder=0)
        axes.axvspan(0, 50, ymin=LLR_cutoff * 105 / 100, ymax=1, facecolor='g', alpha=0.1,zorder=0)
        axes.axvspan(50, 100, ymin=0, ymax=LLR_cutoff * 105 / 100, facecolor='0.5', alpha=0.1,zorder=0)
        axes.axvspan(50, 100, ymin=LLR_cutoff * 105 / 100, ymax=1, facecolor='g', alpha=0.3,zorder=0)
        fig.savefig(output_fig_fn)
        plt.close()