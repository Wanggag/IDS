import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


path_train ='D:/University\ML\dataset\data_balance.csv'
data_set = pd.read_csv(path_train)
x = data_set.iloc[:,1:-1]
columns = x.columns.values
print(columns)
print(data_set.shape)
for i in range(1, 44):
    for j in range (i + 1, 45):
        x = data_set.iloc[:,i]
        y = data_set.iloc[:,j]

        pccs = pearsonr(x, y)
        pccs = np.array(pccs)
        pccs = pccs.astype(np.float32)
        if float(pccs[0]) >= 0.9:
            print('---')
            print(columns[i], columns[j], pccs)

