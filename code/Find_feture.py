from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

model = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)
path_train ='D:/University\ML\dataset\data_balance.csv'
path_test ='D:/University\ML\dataset/test_min_max_scaler.csv'
data_set = pd.read_csv(path_train)
X = data_set.iloc[:,1:-1]
Y = data_set.iloc[:,-1]
model.fit(X, Y)

test_data = pd.read_csv(path_test)
X_test = test_data.iloc[:,1:-1]
Y_test = test_data.iloc[:,-1]
feat_labels = data_set.columns[1:]
predicted = model.predict(X_test)
print(metrics.classification_report(Y_test, predicted))  # 输出结果，精确度、召回率、f-1分数
print(metrics.confusion_matrix(Y_test, predicted))  # 混淆矩阵

auc = metrics.roc_auc_score(Y_test, predicted)
accuracy = metrics.accuracy_score(Y_test, predicted)  # 求精度
print("Accuracy: %.2f%%" % (accuracy * 100.0))

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

