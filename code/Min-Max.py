from sklearn.preprocessing import MinMaxScaler
import pandas as pd
path = 'D:/University\ML\dataset/transform_process.csv'
data_set = pd.read_csv(path)
columns = data_set.columns.values.tolist()
print(columns)
min_max_sacler = MinMaxScaler()
data_set = pd.DataFrame(min_max_sacler.fit_transform(data_set),columns=columns)
data_set.to_csv('D:/University\ML\dataset/min_max_scaler.csv')
