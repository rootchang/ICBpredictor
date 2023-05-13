s############################################################################################################
# Submit batch jobs to server for model hyperparameter search and/or evaluation.
############################################################################################################

import os
import sys

RAND = sys.argv[1]

# MLM_list = ['DecisionTree', 'RandomForest', 'ComplementNaiveBayes', 'MultinomialNaiveBayes', 'GaussianNaiveBayes',
#              'BernoulliNaiveBayes', 'LogisticRegression', 'GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'CatBoost',
#              'LightGBM', 'SupportVectorMachineLinear', 'SupportVectorMachinePoly', 'SupportVectorMachineRadial',
#              'kNearestNeighbourhood', 'DNN', 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3', 'NeuralNetwork4',
#              'GaussianProcess', 'QuadraticDiscriminantAnalysis']  # StandardScaler (#18)
MLM_list = ['TMB', 'LLR6', 'RF6', 'DecisionTree', 'RandomForest', 'LogisticRegression', 'GBoost', 'AdaBoost', 'HGBoost', 'XGBoost',
             'LightGBM', 'SupportVectorMachineRadial',
             'kNearestNeighbourhood', 'DNN', 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3', 'NeuralNetwork4',
             'GaussianProcess']

MLM_list1 = ['TMB', 'RF6', 'DecisionTree', 'RandomForest', 'ComplementNaiveBayes', 'MultinomialNaiveBayes', 'GaussianNaiveBayes',
             'BernoulliNaiveBayes']  # None (#6)
MLM_list2 = ['LLR6', 'LogisticRegression', 'GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'CatBoost', 'LightGBM',
             'SupportVectorMachineLinear', 'SupportVectorMachinePoly', 'SupportVectorMachineRadial',
             'kNearestNeighbourhood', 'DNN', 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3', 'NeuralNetwork4',
             'GaussianProcess', 'QuadraticDiscriminantAnalysis']  # StandardScaler (#18)

TASK = sys.argv[1] # 'PS'  'PE'  'NS'  'NE'

################################################## PanCancer ##################################################
if TASK in ['PS', 'PE']:
    for method in MLM_list:
        if method in MLM_list1:
            data_scale = 'None'
        else:
            data_scale = 'StandardScaler'
        jobNA = method+'_'+data_scale+'_'+RAND+'.run'
        foutNA = 'slurm-'+method+'_'+data_scale+'_'+RAND+'.out'
        command = 'sbatch --job-name='+jobNA+' --output='+foutNA+' --export=TASK='+TASK+',MLM='+method+',RAND='+RAND+' jobscript.sh'
        print(command)
        os.system(command)

################################################## NSCLC ##################################################
if TASK in ['NS', 'NE']:
    for DATA in ['Chowell','DFCI']:
        for method in MLM_list:
            if method in MLM_list1:
                data_scale = 'None'
            else:
                data_scale = 'StandardScaler'
            jobNA = method+'_'+data_scale+'_'+DATA+'_'+RAND+'.run'
            foutNA = 'slurm-'+method+'_'+data_scale+'_'+DATA+'_'+RAND+'.out'
            command = 'sbatch --job-name='+jobNA+' --output='+foutNA+' --export=TASK='+TASK+',MLM='+method+',DATA='+DATA+',RAND='+RAND+' jobscript.sh'
            print(command)
            os.system(command)
