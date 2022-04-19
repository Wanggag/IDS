from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
path_test ='D:/University\ML\dataset/UNSW_NB15_testing-set.csv'

test_data = pd.read_csv(path_test)
proto = test_data[['proto']]
# print(proto.shape)
le = LabelEncoder()
le.fit(proto)
print('class:', le.classes_)
# print((le.classes_).shape)
proto_process = pd.DataFrame(le.transform(proto))
test_data[['proto']] = proto_process

# label11.to_csv('D:/University\ML\dataset/1.csv',header=None)
# label11 = pd.DataFrame(min_max_scaler.fit_transform(label11))
# label11.to_csv('D:/University\ML\dataset/2.csv',header=None)
# print('encode class', le.transform(label1))
# print('min_max_scaler', label11)

service = test_data[['service']]
le.fit(service)
print('class:', le.classes_)
service_process = pd.DataFrame(le.transform(service))
test_data[['service']] = service_process
print(service_process)

state = test_data[['state']]
le.fit(state)
print('class:', le.classes_)
state_process = pd.DataFrame(le.transform(state))
test_data[['state']] = state_process
print(state_process)

attack_cat = test_data[['attack_cat']]
le.fit(attack_cat)
print('class', le.classes_)
attack_cat_process = pd.DataFrame(le.transform(attack_cat))
test_data[['attack_cat']] = attack_cat_process
print(attack_cat_process)

test_data.to_csv('D:/University\ML\dataset/test_transform_process.csv', index=False)

columns = test_data.columns.values.tolist()
min_max_sacler = MinMaxScaler()
test_data = pd.DataFrame(min_max_sacler.fit_transform(test_data),columns=columns)
test_data.to_csv('D:/University\ML\dataset/test_min_max_scaler.csv')