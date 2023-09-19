## Scripts and/or commands for figure reproduction

### 1. Pan-cancer cohort cancer type composition pie plot

`01.cohortCancerComposition_piePlot.Rmd`

### 2. Characterization of patients in different ICB cohorts

`02.ICBcohortsPatientCharacterization_table.Rmd`

### 3. Correlation analysis of features that are measured on a continuous scale

`03.PanCancer_continuousFeatureCorrelation_heatmap.py`

### 4. Pan-cancer logistic regression: feature selection and feature importance

`04.PanCancer_FeaturetImportance.py`

### 5. Pan-cancer model comparison: training and evaluation of different models

`05_1.PanCancer_20Models_HyperParams_Search.py`

`05_2.PanCancer_20Models_evaluation.py`

`05_3.TabNet_paramSearch_evaluation.py`

`05_4.PanCancer_20Models_evaluation_stat.py`

`05_5.PanCancer_LLRx_ParamCalculate.py`

`05_6.PanCancer_LLR6_RF16NBT_Performance_compare.py`

`05_7.PanCancer_LLR6_RF16NBT_AUC_compare.py`

`05_8.PanCancer_LLR6_RF6_TMB_thresholds_on_train.Rmd`

Note: One may use `batchSubmit.py` and `jobscript.sh` to submit batch jobs to server for model hyperparameter search and/or evaluation.

### 6. Pan-cancer external validation of LLR6, RF6 and TMB

`06_1.PanCancer_LLR6_RF6_TMB_multiMetric_compare.py`

`06_2.PanCancer_LLR6_RF6_TMB_ViolinPlot_compare.py`

`06_3.PanCancer_LLR6_vs_LLR6noCancerTerm_ROC_AUC.py`

`06_4.PanCancer_LLR6_vs_LLR5noChemo_ROC_AUC.py`

`06_5.PanCancer_vs_nonNSCLC_LLR6_ROC_AUC.py`

### 7. Pan-cancer prediction of patient survival following ICB or non-ICB treatments by LLR6, monotonic relationship between LORIS and patient ICB response and survival

`07_1.PanCancer_LORIS_TMB_vs_resProb_curve.py`

`07_2.PanCancer_LORIS_TMB_survivalAnalysis_ICB.Rmd`

`07_3.PanCancer_LORIS_ROC_AUC_nonICB_vs_ICB_OS.py`

`07_4.PanCancer_LORIS_TMB_survivalAnalysis_nonICB.Rmd`

`07_5.PanCancer_LORIS_K-Mcurve_individualCancers_ICB.Rmd`


### 8. NSCLC-specific model comparison: training and evaluation of different machine-learning models

`08_1.NSCLC_20Models_HyperParams_Search.py`

`08_2.NSCLC_20Models_evaluation.py`

`08_3.NSCLC_20Models_evaluation_stat.py`

`08_4.NSCLC_LLRx_10k_ParamCalculate.py`

Note: One may use `batchSubmit.py` and `jobscript.sh` to submit batch jobs to server for model hyperparameter search and/or evaluation.

### 9. NSCLC-specific external validation of LLR6

`09_1.NSCLC_LLR6_PDL1_TMB_multiMetric_compare.py`

`09_2.NSCLC_LLR6_PDL1_TMB_ViolinPlot_compare.py`

`09_3.NSCLC_vs_PanCancer_LLR6_ROC_AUC.py`

`09_4.NSCLC_LLR6_vs_LLR2_ROC_AUC.py`

`09_5.NSCLC_LLR6_vs_LLR5noChemo_ROC_AUC.py`

### 10. Hazard ratio prediction of patient survival following ICB by NSCLC-specific LLR6

`10.NSCLC_LORIS_PDL1_TMB_ForestPlot_individualDatasets.Rmd`


### 11. Predictive power of NSCLC-specific LLR6 model in other cancer types

`11.NSCLC_LLR6_on_otherCancers_ROC_AUC.py`

### 12. Pan-cancer formula for LORIS calculation

`12.Formula_LORIS.py`