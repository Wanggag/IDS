from collections import Counter
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
path = 'D:/University\ML\dataset\min_max_scaler.csv'
data_set = pd.read_csv(path)
print(data_set.groupby(['label'])['label'].count())
columns = data_set.columns.values.tolist()
ada = ADASYN(random_state=42)
X = data_set.iloc[:,0:43]
Y = data_set.iloc[:,-1]
# print(X)
# print(Y)
X_resampled, y_resampled = ada.fit_resample(X, Y)
print(X_resampled.shape)
print(y_resampled.shape)
X_resampled['label'] = y_resampled
data_balance = pd.DataFrame(X_resampled,columns=columns)
data_balance.to_csv('D:/University\ML\dataset\data_balance.csv')
print(sorted(Counter(y_resampled).items()))
