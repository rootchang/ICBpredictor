## Scripts and/or commands for figure reproduction

### 0. Cancer type composition pie plot

`00.cancerType_piePlot.Rmd`

### 1. Harmonization of TMB/NLR measured with different methods

`01.TMB_NLR_harmonization.Rmd`

### 2. Statistics of characteristics of patients in the study

`01.Table1_stats.Rmd`

### 3. Correlation analysis of features that are measured on a continuous scale

`03.PanCancer_FeatureCor.py`


### 4. Pan-cancer logistic regression: feature selection and feature importance

`04.PanCancer_LR_HyperParamsSearch_FeatImportance.py`

### 5. Pan-cancer model comparison: training and evaluation of different models

`05_1.PanCancer_20Models_HyperParams_Search.py`

`05_2.PanCancer_20Models_evaluation.py`

`05_3.PanCancer_20Models_evaluation_plot.py`

`05_4.PanCancer_LLR6_10k_ParamCalculate.py`

`05_5.PanCancer_LLR6_RF6_TMB_ROC_train_plot.py`

`05_6.PanCancer_LLR6_RF6_TMB_Cutoff_train_plot.Rmd`

`05_7.PanCancer_LLR6_RF6_TMB_evaluation_plot.py`

Note: One may use `batchSubmit.py` and `jobscript.sh` to submit batch jobs to server for model hyperparameter search and/or evaluation.

### 6. Pan-cancer external validation of LLR6, RF6 and TMB

`06_1.PanCancer_LLR6_RF6_TMB_All_multiMetric_plot.py`

`06_2.PanCancer_LLR6_RF6_TMB_All_ViolinPlot.py`


### 7. Pan-cancer prediction of patient survival following ICB by LLR6, monotonic relationship between LORIS and patient ICB response

`07.PanCancer_LLR6_TMB_survivalAnalysis.Rmd`


### 8. NSCLC-specific model comparison: training and evaluation of different machine-learning models

`08_1.NSCLC_20Models_HyperParams_Search.py`

`08_2.NSCLC_20Models_evaluation.py`

`08_3.NSCLC_20Models_evaluation_plot.py`

`08_4.NSCLC_LLR6_10k_ParamCalculate.py`

`08_5.NSCLC_LLR6_PDL1_TMB_ROC_OR_train_plot.py`

Note: One may use `batchSubmit.py` and `jobscript.sh` to submit batch jobs to server for model hyperparameter search and/or evaluation.

### 9. NSCLC-specific external validation of LLR6

`09_1.NSCLC_LLR6_PDL1_TMB_All_multiMetric_plot.py`

`09_2.NSCLC_LLR6_PDL1_TMB_All_ViolinPlot.py`

### 10. NSCLC-specific prediction of patient survival following ICB by LLR6

`10.NSCLC_LLR6_PDL1_TMB_ForestPlot.Rmd`


### 11. Predictive power of NSCLC-specific LLR6 model for other cancer types

`13.NSCLCmodel_on_otherCancers_multiMetric_plot.py`

### 12. Pan-cancer formula for LORIS calculation

`12.Formula_LORIS.py`