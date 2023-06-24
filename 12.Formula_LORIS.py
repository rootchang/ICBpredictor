import numpy as np

if __name__ == "__main__":

    ########################## Order of features ##########################
    # featuresNA = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
    #               'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
    #               'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
    #               'CancerType14', 'CancerType15', 'CancerType16'] # pan-cancer feature order

    ###################### Read in LLRx model params ######################
    fnIn = '../03.Results/6features/PanCancer/PanCancer_LLR6_10k_ParamCalculate.txt'
    params_data = open(fnIn, 'r').readlines()
    params_dict = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict[param_name] = params_val

    ########################## calculate LORIS formula ##########################
    scaler_mean_ = np.array(params_dict['LLR_mean'])
    scaler_scale_ = np.array(params_dict['LLR_scale'])
    clf_coef_ = np.array([params_dict['LLR_coef']])
    clf_intercept_ = np.array(params_dict['LLR_intercept'])

    coef_list = params_dict['LLR_coef']/scaler_scale_
    print('merged coef: ', [round(c,4) for c in coef_list])
    interc = -sum(params_dict['LLR_coef']*scaler_mean_/scaler_scale_) + params_dict['LLR_intercept'][0]
    print('merge intercept: ', round(interc,4))
