import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas.plotting import table


def table_gen(reports, latex=False, tablestxt='', model_name=False):
    df = pd.DataFrame(reports)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    fig, ax = plt.subplots()
    tablee = table(ax, df, loc='center', colWidths=[0.2] * len(df.columns), cellLoc='center')

    # Bold Col names
    for i in range(len(df.columns)):
        tablee[0, i].get_text().set_weight('bold')

    # Bold col max values
    max_values = df.iloc[:, 1:].max()
    min_values = df.iloc[:, 1:].min()
    for i, max_value in enumerate(max_values):
        col_max_idx = df.iloc[:, 1:].iloc[:, i].idxmax()
        tablee[col_max_idx + 1, i + 1].get_text().set_weight('bold')
    for j, mix_value in enumerate(min_values):
        col_min_idx = df.iloc[:, 1:].iloc[:, j].idxmin()
        tablee[col_min_idx + 1, j + 1].get_text().set_weight('bold')

    # Generate latex code
    if latex:
        latex_code = ''
        if model_name:
                latex_code = df.to_latex(index=False)
        else:
            latex_code = df.iloc[:, 1:].to_latex(index=False)

        if os.path.isfile(f'ML_models/new_data/reports/{tablestxt}.txt'):
            with open(f'ML_models/new_data/reports/{tablestxt}.txt', 'a') as file:
                file.write(f'\n{tablestxt} file:\n')
                file.write(latex_code)
        else:
            with open(f'ML_models/new_data/reports/{tablestxt}.txt', 'w') as file:
                file.write(f'{tablestxt} file:\n')
                file.write(latex_code)

    return