from pathlib import Path
from ML_models import Classification as clss
from ML_models import Regression as regress


# Data paths
data_path = 'EDA_ETL/Features/new_data/df_marks.csv'

# report = clss.cross_valid_models(data_path, 'Random Forest')
# report = clss.qda_model(data_path)
# report = regress.ridge_regress(data_path)
report = regress.knn_regress(data_path, 'Decision Tree')

# print(report)