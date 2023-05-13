import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

plt.rcParams.update({'font.size': 9})
plt.rcParams["font.family"] = "Arial"
palette = sns.color_palette("deep")

if __name__ == "__main__":

    bs_number = 1000 # multi-time resampling for mean,sd,CI
    random.seed(1)

    Plot_type = sys.argv[1] # 'NSCLC'  'PanCancer'
    LLRmodelNA = sys.argv[2] # 'LLR6'  'LLRnoTMB'   'LLRnoChemo'

    start_time = time.time()
    print('Raw data read in ...')
    if Plot_type=='NSCLC':
        fnIn = '../03.Results/NSCLC_'+LLRmodelNA+'_Scaler(StandardScaler)_prediction.xlsx'
    else:
        fnIn = '../03.Results/PanCancer_' + LLRmodelNA+'_Scaler(StandardScaler)_prediction.xlsx'
    y_pred_LLR = []
    y_true = []
    start_set = 1
    if Plot_type=='NSCLC':
        end_set = 4
    else:
        end_set = 3
    output_curve_fn = '../03.Results/'+LLRmodelNA+'_LORIS_vs_ORR_'+Plot_type+'.pdf'
    for sheet_i in range(start_set,end_set): # range(start_set,3)
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_LLR.extend(data['y_pred'].tolist())
        y_true.extend(data['y'].tolist())

    if Plot_type=='NSCLC':
        fnIn = '../03.Results/NSCLC_TMB_Scaler(None)_prediction.xlsx'
    else:
        fnIn = '../03.Results/PanCancer_TMB_Scaler(None)_prediction.xlsx'
    y_pred_TMB = []
    for sheet_i in range(start_set,end_set):
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_TMB.extend(data['y_pred'].tolist())

    if Plot_type=='NSCLC':
        fnIn = '../03.Results/NSCLC_PDL1_Scaler(None)_prediction.xlsx'
        y_pred_PDL1 = []
        for sheet_i in range(start_set,end_set):
            data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
            y_pred_PDL1_temp = data['y_pred']/100
            y_pred_PDL1.extend(y_pred_PDL1_temp.tolist())
    else:
        fnIn = '../03.Results/PanCancer_RF6_Scaler(None)_prediction.xlsx'
        y_pred_RF6 = []
        for sheet_i in range(start_set,end_set):
            data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
            y_pred_RF6_temp = data['y_pred']
            y_pred_RF6.extend(y_pred_RF6_temp.tolist())

    y_true = np.array(y_true)
    y_pred_LLR = np.array(y_pred_LLR)
    if Plot_type == 'NSCLC':
        y_pred_PDL1 = np.array(y_pred_PDL1)
    else:
        y_pred_RF6 = np.array(y_pred_RF6)
    y_pred_TMB = np.array(y_pred_TMB)
    score_list_LLR = np.arange(0.0, 1.01, 0.01)
    score_list_TMB = np.arange(0, 101, 1)
    LLR_num = len(score_list_LLR)
    TMB_num = len(score_list_TMB)
    if Plot_type == 'NSCLC':
        score_list_PDL1 = np.arange(0.0, 1.01, 0.01)
        PDL1_num = len(score_list_PDL1)
    else:
        score_list_RF6 = np.arange(0.0, 1.01, 0.01)
        RF6_num = len(score_list_RF6)

    LLRhigh_ORR_list = [[] for _ in range(LLR_num)]
    TMBhigh_ORR_list = [[] for _ in range(TMB_num)]
    LLR_ORR_list = [[] for _ in range(LLR_num)]
    TMB_ORR_list = [[] for _ in range(TMB_num)]
    LLRlow_ORR_list = [[] for _ in range(LLR_num)]
    TMBlow_ORR_list = [[] for _ in range(TMB_num)]

    LLR_patientNUM_list = [[] for _ in range(LLR_num)]
    TMB_patientNUM_list = [[] for _ in range(TMB_num)]
    sampleNUM = len(y_true)
    idx_list = range(sampleNUM)
    print('Sample num:',sampleNUM)

    if Plot_type == 'NSCLC':
        PDL1high_ORR_list = [[] for _ in range(PDL1_num)]
        PDL1_ORR_list = [[] for _ in range(PDL1_num)]
        PDL1low_ORR_list = [[] for _ in range(PDL1_num)]
        PDL1_patientNUM_list = [[] for _ in range(PDL1_num)]
    else:
        RF6high_ORR_list = [[] for _ in range(RF6_num)]
        RF6_ORR_list = [[] for _ in range(RF6_num)]
        RF6low_ORR_list = [[] for _ in range(RF6_num)]
        RF6_patientNUM_list = [[] for _ in range(RF6_num)]

    for bs in range(bs_number):
        idx_resampled = random.choices(idx_list, k = sampleNUM)
        y_true_resampled = y_true[idx_resampled]
        y_pred_LLR_resampled = y_pred_LLR[idx_resampled]
        y_pred_TMB_resampled = y_pred_TMB[idx_resampled]
        if Plot_type == 'NSCLC':
            y_pred_PDL1_resampled = y_pred_PDL1[idx_resampled]
        else:
            y_pred_RF6_resampled = y_pred_RF6[idx_resampled]
        for score_i in range(len(score_list_LLR)):
            score = score_list_LLR[score_i]
            idx_high_interval = y_pred_LLR_resampled >= score
            y_true_high = y_true_resampled[idx_high_interval]
            Rhigh_num = sum(y_true_high)
            tot_high_num = len(y_true_high)
            patientRatio_temp = sum(y_pred_LLR_resampled < score) / sampleNUM
            LLR_patientNUM_list[score_i].append(patientRatio_temp)
            if not tot_high_num:
                LLRhigh_ORR_list[score_i].append(LLRhigh_ORR_list[score_i-1][-1])
            else:
                ORRhigh_temp = Rhigh_num / tot_high_num
                LLRhigh_ORR_list[score_i].append(ORRhigh_temp)

            idx_low_interval = y_pred_LLR_resampled < score
            y_true_low = y_true_resampled[idx_low_interval]
            Rlow_num = sum(y_true_low)
            tot_low_num = len(y_true_low)
            if not tot_low_num:
                LLRlow_ORR_list[score_i].append(0)
            else:
                ORRlow_temp = Rlow_num / tot_low_num
                LLRlow_ORR_list[score_i].append(ORRlow_temp)

            if sum(y_pred_LLR_resampled <= score+0.05) < 0.01*len(y_pred_LLR_resampled): # skip
                idx_interval = []
            elif sum(y_pred_LLR_resampled > score-0.05) < 0.01*len(y_pred_LLR_resampled): # merge patients
                idx_interval = (y_pred_LLR_resampled > score-0.05)
            else:
                idx_interval = (y_pred_LLR_resampled <= score+0.05) & (y_pred_LLR_resampled > score-0.05)
            y_true_temp = y_true_resampled[idx_interval]
            R_num = sum(y_true_temp)
            tot_num = len(y_true_temp)
            if not tot_num:
                LLR_ORR_list[score_i].append(0)
            else:
                ORR_temp = R_num / tot_num
                LLR_ORR_list[score_i].append(ORR_temp)
            if sum(y_pred_LLR_resampled > score - 0.05) < 0.01 * len(y_pred_LLR_resampled):# finish score_list_LLR loop
                break
        for score_i in range(len(score_list_TMB)):
            score = score_list_TMB[score_i]
            idx_high_interval = y_pred_TMB_resampled >= score
            y_true_high = y_true_resampled[idx_high_interval]
            Rhigh_num = sum(y_true_high)
            tot_high_num = len(y_true_high)
            patientRatio_temp = sum(y_pred_TMB_resampled < score) / sampleNUM
            TMB_patientNUM_list[score_i].append(patientRatio_temp)
            if not tot_high_num:
                TMBhigh_ORR_list[score_i].append(TMBhigh_ORR_list[score_i - 1][-1])
            else:
                ORRhigh_temp = Rhigh_num / tot_high_num
                TMBhigh_ORR_list[score_i].append(ORRhigh_temp)

            idx_low_interval = y_pred_TMB_resampled < score
            y_true_low = y_true_resampled[idx_low_interval]
            Rlow_num = sum(y_true_low)
            tot_low_num = len(y_true_low)
            if not tot_low_num:
                TMBlow_ORR_list[score_i].append(0)
            else:
                ORRlow_temp = Rlow_num / tot_low_num
                TMBlow_ORR_list[score_i].append(ORRlow_temp)

            if sum(y_pred_TMB_resampled <= score + 5) < 0.01 * len(y_pred_TMB_resampled):  # skip
                idx_interval = []
            elif sum(y_pred_TMB_resampled > score - 5) < 0.01 * len(y_pred_TMB_resampled):  # merge patients
                idx_interval = (y_pred_TMB_resampled > score - 5)
            else:
                idx_interval = (y_pred_TMB_resampled <= score + 5) & (y_pred_TMB_resampled > score - 5)
            y_true_temp = y_true_resampled[idx_interval]
            R_num = sum(y_true_temp)
            tot_num = len(y_true_temp)
            if not tot_num:
                TMB_ORR_list[score_i].append(0)
            else:
                ORR_temp = R_num / tot_num
                TMB_ORR_list[score_i].append(ORR_temp)
            if sum(y_pred_TMB_resampled > score - 5) < 0.01 * len(y_pred_TMB_resampled):  # finish score_list_LLR loop
                break
        if Plot_type == 'NSCLC':
            for score_i in range(len(score_list_PDL1)):
                score = score_list_PDL1[score_i]
                idx_high_interval = y_pred_PDL1_resampled >= score
                y_true_high = y_true_resampled[idx_high_interval]
                Rhigh_num = sum(y_true_high)
                tot_high_num = len(y_true_high)
                patientRatio_temp = sum(y_pred_PDL1_resampled < score) / sampleNUM
                PDL1_patientNUM_list[score_i].append(patientRatio_temp)
                if not tot_high_num:
                    PDL1high_ORR_list[score_i].append(PDL1high_ORR_list[score_i - 1][-1])
                else:
                    ORRhigh_temp = Rhigh_num / tot_high_num
                    PDL1high_ORR_list[score_i].append(ORRhigh_temp)

                idx_low_interval = y_pred_PDL1_resampled < score
                y_true_low = y_true_resampled[idx_low_interval]
                Rlow_num = sum(y_true_low)
                tot_low_num = len(y_true_low)
                if not tot_low_num:
                    PDL1low_ORR_list[score_i].append(0)
                else:
                    ORRlow_temp = Rlow_num / tot_low_num
                    PDL1low_ORR_list[score_i].append(ORRlow_temp)

                if sum(y_pred_PDL1_resampled <= score + 0.05) < 0.01 * len(y_pred_PDL1_resampled):  # skip
                    idx_interval = []
                elif sum(y_pred_PDL1_resampled > score - 0.05) < 0.01 * len(y_pred_PDL1_resampled):  # merge patients
                    idx_interval = (y_pred_PDL1_resampled > score - 0.05)
                else:
                    idx_interval = (y_pred_PDL1_resampled <= score + 0.05) & (y_pred_PDL1_resampled > score - 0.05)
                y_true_temp = y_true_resampled[idx_interval]
                R_num = sum(y_true_temp)
                tot_num = len(y_true_temp)
                if not tot_num:
                    PDL1_ORR_list[score_i].append(0)
                else:
                    ORR_temp = R_num / tot_num
                    PDL1_ORR_list[score_i].append(ORR_temp)
                if sum(y_pred_PDL1_resampled > score - 0.05) < 0.01 * len(y_pred_PDL1_resampled):# finish score_list_LLR loop
                    break
        else:
            for score_i in range(len(score_list_RF6)):
                score = score_list_RF6[score_i]
                idx_high_interval = y_pred_RF6_resampled >= score
                y_true_high = y_true_resampled[idx_high_interval]
                Rhigh_num = sum(y_true_high)
                tot_high_num = len(y_true_high)
                patientRatio_temp = sum(y_pred_RF6_resampled < score) / sampleNUM
                RF6_patientNUM_list[score_i].append(patientRatio_temp)
                if not tot_high_num:
                    RF6high_ORR_list[score_i].append(RF6high_ORR_list[score_i - 1][-1])
                else:
                    ORRhigh_temp = Rhigh_num / tot_high_num
                    RF6high_ORR_list[score_i].append(ORRhigh_temp)

                idx_low_interval = y_pred_RF6_resampled < score
                y_true_low = y_true_resampled[idx_low_interval]
                Rlow_num = sum(y_true_low)
                tot_low_num = len(y_true_low)
                if not tot_low_num:
                    RF6low_ORR_list[score_i].append(0)
                else:
                    ORRlow_temp = Rlow_num / tot_low_num
                    RF6low_ORR_list[score_i].append(ORRlow_temp)

                if sum(y_pred_RF6_resampled <= score + 0.05) < 0.01 * len(y_pred_RF6_resampled):  # skip
                    idx_interval = []
                elif sum(y_pred_RF6_resampled > score - 0.05) < 0.01 * len(y_pred_RF6_resampled):  # merge patients
                    idx_interval = (y_pred_RF6_resampled > score - 0.05)
                else:
                    idx_interval = (y_pred_RF6_resampled <= score + 0.05) & (y_pred_RF6_resampled > score - 0.05)
                y_true_temp = y_true_resampled[idx_interval]
                R_num = sum(y_true_temp)
                tot_num = len(y_true_temp)
                if not tot_num:
                    RF6_ORR_list[score_i].append(0)
                else:
                    ORR_temp = R_num / tot_num
                    RF6_ORR_list[score_i].append(ORR_temp)
                if sum(y_pred_RF6_resampled > score - 0.05) < 0.01 * len(
                        y_pred_RF6_resampled):  # finish score_list_LLR loop
                    break
    # remove empty elements (upon scores that near 1)
    for i in range(len(LLRhigh_ORR_list)):
        if len(LLRhigh_ORR_list[i])==0:
            break
    LLRhigh_ORR_list = LLRhigh_ORR_list[0:i]
    LLRlow_ORR_list = LLRlow_ORR_list[0:i]
    LLR_ORR_list = LLR_ORR_list[0:i]
    LLR_patientNUM_list = LLR_patientNUM_list[0:i]
    score_list_LLR = score_list_LLR[0:i]
    LLRhigh_ORR_mean = [np.mean(c) for c in LLRhigh_ORR_list]
    LLRhigh_ORR_05 = [np.quantile(c,0.05) for c in LLRhigh_ORR_list]
    LLRhigh_ORR_95 = [np.quantile(c,0.95) for c in LLRhigh_ORR_list]
    LLRlow_ORR_mean = [np.mean(c) for c in LLRlow_ORR_list]
    LLRlow_ORR_05 = [np.quantile(c, 0.05) for c in LLRlow_ORR_list]
    LLRlow_ORR_95 = [np.quantile(c, 0.95) for c in LLRlow_ORR_list]
    LLRlow_patientRatio_mean = [np.mean(c) for c in LLR_patientNUM_list]
    LLR_ORR_mean = [np.mean(c) for c in LLR_ORR_list]
    LLR_ORR_05 = [np.quantile(c, 0.05) for c in LLR_ORR_list]
    LLR_ORR_95 = [np.quantile(c, 0.95) for c in LLR_ORR_list]
    LLR_patientRatio_mean = [np.mean(c) for c in LLR_patientNUM_list]
    print('LLR response odds:')
    for i in range(len(LLRhigh_ORR_95)):
        print(score_list_LLR[i], LLRhigh_ORR_mean[i], LLRlow_ORR_mean[i], LLR_ORR_mean[i], LLRlow_patientRatio_mean[i])

    # remove empty elements (upon scores that near 1)
    for i in range(len(TMBhigh_ORR_list)):
        if len(TMBhigh_ORR_list[i]) == 0:
            break
    TMBhigh_ORR_list = TMBhigh_ORR_list[0:i]
    TMBlow_ORR_list = TMBlow_ORR_list[0:i]
    TMB_ORR_list = TMB_ORR_list[0:i]
    TMB_patientNUM_list = TMB_patientNUM_list[0:i]
    score_list_TMB = score_list_TMB[0:i]
    TMBhigh_ORR_mean = [np.mean(c) for c in TMBhigh_ORR_list]
    TMBhigh_ORR_05 = [np.quantile(c, 0.05) for c in TMBhigh_ORR_list]
    TMBhigh_ORR_95 = [np.quantile(c, 0.95) for c in TMBhigh_ORR_list]
    TMBlow_ORR_mean = [np.mean(c) for c in TMBlow_ORR_list]
    TMBlow_ORR_05 = [np.quantile(c, 0.05) for c in TMBlow_ORR_list]
    TMBlow_ORR_95 = [np.quantile(c, 0.95) for c in TMBlow_ORR_list]
    TMBlow_patientRatio_mean = [np.mean(c) for c in TMB_patientNUM_list]
    TMB_ORR_mean = [np.mean(c) for c in TMB_ORR_list]
    TMB_ORR_05 = [np.quantile(c, 0.05) for c in TMB_ORR_list]
    TMB_ORR_95 = [np.quantile(c, 0.95) for c in TMB_ORR_list]
    TMB_patientRatio_mean = [np.mean(c) for c in TMB_patientNUM_list]
    print('TMB response odds:')
    for i in range(len(TMBhigh_ORR_95)):
        print(score_list_TMB[i], TMBhigh_ORR_mean[i], TMBlow_ORR_mean[i], TMB_ORR_mean[i], TMBlow_patientRatio_mean[i])

    if Plot_type == 'NSCLC':
        # remove empty elements (upon scores that near 1)
        for i in range(len(PDL1high_ORR_list)):
            if len(PDL1high_ORR_list[i]) == 0:
                break
        PDL1high_ORR_list = PDL1high_ORR_list[0:i]
        PDL1low_ORR_list = PDL1low_ORR_list[0:i]
        PDL1_ORR_list = PDL1_ORR_list[0:i]
        PDL1_patientNUM_list = PDL1_patientNUM_list[0:i]
        score_list_PDL1 = score_list_PDL1[0:i]
        PDL1high_ORR_mean = [np.mean(c) for c in PDL1high_ORR_list]
        PDL1high_ORR_05 = [np.quantile(c, 0.05) for c in PDL1high_ORR_list]
        PDL1high_ORR_95 = [np.quantile(c, 0.95) for c in PDL1high_ORR_list]
        PDL1low_ORR_mean = [np.mean(c) for c in PDL1low_ORR_list]
        PDL1low_ORR_05 = [np.quantile(c, 0.05) for c in PDL1low_ORR_list]
        PDL1low_ORR_95 = [np.quantile(c, 0.95) for c in PDL1low_ORR_list]
        PDL1low_patientRatio_mean = [np.mean(c) for c in PDL1_patientNUM_list]
        PDL1_ORR_mean = [np.mean(c) for c in PDL1_ORR_list]
        PDL1_ORR_05 = [np.quantile(c, 0.05) for c in PDL1_ORR_list]
        PDL1_ORR_95 = [np.quantile(c, 0.95) for c in PDL1_ORR_list]
        PDL1_patientRatio_mean = [np.mean(c) for c in PDL1_patientNUM_list]
        print('PDL1 response odds:')
        for i in range(len(PDL1high_ORR_95)):
            print(score_list_PDL1[i], PDL1high_ORR_mean[i], PDL1low_ORR_mean[i], PDL1_ORR_mean[i], PDL1low_patientRatio_mean[i])
    else:
        # remove empty elements (upon scores that near 1)
        for i in range(len(RF6high_ORR_list)):
            if len(RF6high_ORR_list[i]) == 0:
                break
        RF6high_ORR_list = RF6high_ORR_list[0:i]
        RF6low_ORR_list = RF6low_ORR_list[0:i]
        RF6_ORR_list = RF6_ORR_list[0:i]
        RF6_patientNUM_list = RF6_patientNUM_list[0:i]
        score_list_RF6 = score_list_RF6[0:i]
        RF6high_ORR_mean = [np.mean(c) for c in RF6high_ORR_list]
        RF6high_ORR_05 = [np.quantile(c, 0.05) for c in RF6high_ORR_list]
        RF6high_ORR_95 = [np.quantile(c, 0.95) for c in RF6high_ORR_list]
        RF6low_ORR_mean = [np.mean(c) for c in RF6low_ORR_list]
        RF6low_ORR_05 = [np.quantile(c, 0.05) for c in RF6low_ORR_list]
        RF6low_ORR_95 = [np.quantile(c, 0.95) for c in RF6low_ORR_list]
        RF6low_patientRatio_mean = [np.mean(c) for c in RF6_patientNUM_list]
        RF6_ORR_mean = [np.mean(c) for c in RF6_ORR_list]
        RF6_ORR_05 = [np.quantile(c, 0.05) for c in RF6_ORR_list]
        RF6_ORR_95 = [np.quantile(c, 0.95) for c in RF6_ORR_list]
        RF6_patientRatio_mean = [np.mean(c) for c in RF6_patientNUM_list]
        print('RF6 response odds:')
        for i in range(len(RF6high_ORR_95)):
            print(score_list_RF6[i], RF6high_ORR_mean[i], RF6low_ORR_mean[i], RF6_ORR_mean[i],
                  RF6low_patientRatio_mean[i])

    ############# Score-Prob curve ##############
    subplot_num = 3
    fig1, axes = plt.subplots(1, subplot_num, figsize=(6.4, 1.7))
    fig1.subplots_adjust(left=0.1, bottom=0.22, right=0.98, top=0.96, wspace=0.45, hspace=0.45)

    axes[0].plot(score_list_LLR, LLR_ORR_mean, '-', color='r')
    axes[0].fill_between(score_list_LLR, LLR_ORR_05, LLR_ORR_95, facecolor='r', alpha=0.25)
    axes[0].set_ylabel("Response probability (%)", color="k")
    axes[0].set_xlabel('LORIS') # LORIS (LLR)

    if Plot_type == 'NSCLC':
        axes[1].plot(score_list_PDL1, PDL1_ORR_mean, '-', color='r')
        axes[1].fill_between(score_list_PDL1, PDL1_ORR_05, PDL1_ORR_95, facecolor='r', alpha=0.25)
        axes[1].set_ylabel("Response probability (%)", color="k")
        axes[1].set_xlabel('PD-L1 TPS (%)')
    else:
        axes[1].plot(score_list_RF6, RF6_ORR_mean, '-', color='r')
        axes[1].fill_between(score_list_RF6, RF6_ORR_05, RF6_ORR_95, facecolor='r', alpha=0.25)
        axes[1].set_ylabel("Response probability (%)", color="k")
        axes[1].set_xlabel('RF6 score')

    axes[2].plot(score_list_TMB, TMB_ORR_mean, '-', color='r')
    axes[2].fill_between(score_list_TMB, TMB_ORR_05, TMB_ORR_95, facecolor='r', alpha=0.25)
    axes[2].set_ylabel("Response probability (%)", color="k")
    axes[2].set_xlabel('TMB')


    for j in range(subplot_num):
        axes[j].set_ylim([-0.02, 1.02])
        axes[j].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes[j].set_yticklabels([0, 25, 50, 75, 100])
        axes[j].spines['right'].set_visible(False)
        axes[j].spines['top'].set_visible(False)

    for j in range(subplot_num-1):
        axes[j].set_xlim([0, 1])
        axes[j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axes[j].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    if Plot_type=='NSCLC':
        axes[1].set_xticklabels([0, 20, 40, 60, 80, 100])
    axes[2].set_xlim([0, 53])
    axes[2].set_xticks([0, 10, 20, 30,40,50])

    if Plot_type != 'NSCLC':
        if LLRmodelNA == 'LLR5noChemo':
            axes[0].axvspan(0, 0.28, facecolor='k', alpha=0.1)
            axes[0].axvspan(0.75, 1, facecolor='g', alpha=0.1)
            axes[2].axvspan(27, 101, facecolor='g', alpha=0.1)
        elif LLRmodelNA == 'LLR6':
            axes[0].axvspan(0, 0.275, facecolor='k', alpha=0.1)
            axes[0].axvspan(0.695, 1, facecolor='g', alpha=0.1)
            axes[1].axvspan(0.595, 1, facecolor='g', alpha=0.1)
            axes[2].axvspan(26.5, 101, facecolor='g', alpha=0.1)
    else:
        if LLRmodelNA == 'LLR5noChemo':
            axes[0].axvspan(0.74, 1, facecolor='g', alpha=0.1)
            axes[1].axvspan(0.95, 1, facecolor='g', alpha=0.1)
        elif LLRmodelNA == 'LLR6':
            axes[0].axvspan(0, 0.36, facecolor='k', alpha=0.1)
            axes[0].axvspan(0.72, 1, facecolor='g', alpha=0.1)
            axes[1].axvspan(0.95, 1, facecolor='g', alpha=0.1)
    plt.savefig(output_curve_fn) # , dpi=300
    plt.close()