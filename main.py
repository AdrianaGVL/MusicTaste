from pathlib import Path
from ML_models import Classification as clss
from ML_models import Regression as regress
from ML_models import reports as repo


# Data
# Dataset for classification purposes
cdata_path = 'EDA_ETL/Features/new_data/df_marks.csv'
# Dataset for regression purposes
rdata_path = 'EDA_ETL/Features/new_data/df_energy.csv'
# Normalize dataset for regression purposes
rdata_norm_path = 'EDA_ETL/Features/new_data/df_enorm.csv'

# Models
classificationModels = ['QDA', 'Classification Tree', 'Random Forest']
regressionModels = ['Ridge', 'Regression Tree', 'KNN']

# R is for regression | C is for classification
Application = 'C'

if Application == 'C':
    # Classification
    for i in range (len(classificationModels)):
        if classificationModels[i] == 'QDA':
            print(f'{classificationModels[i]} Model:')
            report = clss.qda_model(cdata_path)
            print('\n')
            repo.table_gen(report, latex=True, tablestxt='QDA_model')
        else:
            print(f'{classificationModels[i]} Model:')
            report = clss.cross_valid_models(cdata_path, classificationModels[i])
            print('\n')
            repo.table_gen(report, latex=True, tablestxt=f'{classificationModels[i]}_model')

else:
    # Regression
    for i in range (len(regressionModels)):
        if regressionModels[i] == 'Ridge':
            print(f'{regressionModels[i]} Model:')
            report = regress.ridge_regress(rdata_norm_path)
            print('\n')
            repo.table_gen(report, latex=True, tablestxt='Ridge_model')
        else:
            print(f'{regressionModels[i]} Model:')
            report = regress.cross_validation(rdata_path, regressionModels[i])
            print('\n')
            repo.table_gen(report, latex=True, tablestxt=f'{regressionModels[i]}_model')