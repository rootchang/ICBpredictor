## Scripts and/or commands for figure reproduction

### 0. TMB harmonization

`00.TMB_harmonization.Rmd`

### 1. Statistics of characteristics of patients in the study

`01.Table1_stats.Rmd`

### 2. Correlation analysis of features that are measured on a continuous scale

#### 2.1 Pan-cancer

`02_1.PanCancer_FeatureCor.py`

#### 2.2 NSCLC

`02_2.NSCLC_FeatureCor.py`

### 3. Predictive power of individual features on ICB response

#### 3.1 Pan-cancer

`03_1.PanCancer_SingleFeaturePredPower.py`

#### 3.2 NSCLC

`03_2.NSCLC_SingleFeaturePredPower.py`

### 4. Pan-cancer logistic regression: feature selection and feature importance

`04.PanCancer_LR_HyperParamsSearch_FeatImportance.py`

### 5. Pan-cancer model comparison: training and evaluation of different models

`05_1.PanCancer_20Models_HyperParams_Search.py`

`05_2.PanCancer_20Models_evaluation.py`

`05_3.PanCancer_20Models_evaluation_plot.py`

`05_4.PanCancer_LLRx_10k_ParamCalculate.py`

`05_5.PanCancer_LLRx_RF6_TMB_ROC_train_plot.py`

`05_6.PanCancer_LLRx_RF6_TMB_Cutoff_train_plot.Rmd`

`05_7.PanCancer_LLRx_RF6_TMB_evaluation_plot.py`

Note: One may use `batchSubmit.py` and `jobscript.sh` to submit batch jobs to server for model hyperparameter search and/or evaluation.

### 6. Pan-cancer external validation of LLRx, RF6 and TMB

`06_1.PanCancer_LLRx_RF6_TMB_Test_multiMetric_plot.py`

`06_2.PanCancer_LLRx_RF6_TMB_Test_ViolinPlot.py`


### 7. Pan-cancer prediction of patient survival following ICB by LLRx

`07.PanCancer_LLRx_TMB_KMcurve.Rmd`


### 8. NSCLC-specific model comparison: training and evaluation of 17 machine-learning models

`08_1.NSCLC_20Models_HyperParams_Search.py`

`08_2.NSCLC_20Models_evaluation.py`

`08_3.NSCLC_20Models_evaluation_plot.py`

`08_4.NSCLC_LLRx_10k_ParamCalculate.py`

`08_5.NSCLC_LLRx_PDL1_TMB_ROC_OR_train_plot.py`

Note: One may use `batchSubmit.py` and `jobscript.sh` to submit batch jobs to server for model hyperparameter search and/or evaluation.

### 9. NSCLC-specific external validation of LLR6

`09_1.NSCLC_LLRx_TMB_PDL1_Test_multiMetric_plot.py`

`09_2.NSCLC_LLRx_RF6_TMB_Test_ViolinPlot.py`

### 10. NSCLC-specific prediction of patient survival following ICB by LLR6

`10.NSCLC_LLRx_TMB_PDL1_corrAnalysis_KMcurve.Rmd`

### 11. Pan-cancer & NSCLC-specific scatter distribution of responders and non-responders versus LLR & TMB or PDL1 score

`11.LLRx_TMB_scatterPlot.py`

### 12. Pan-cancer & NSCLC-specific monotonic relationship between LORIS and patient ICB response

`12_1.LLRx_TMB_Score_vs_ORR_plot.py`

`12_2.LLRx_TMB_Monotonic_survivalAnalysis.Rmd`

### 13. Predictive value for ICB response by adding PD-L1 in other cancer types

`13.NSCLCmodel_on_otherCancers_multiMetric_plot.py`

### 14. Pan-cancer & NSCLC-specific formula for LORIS calculation

`14.Formula_LORIS.py`