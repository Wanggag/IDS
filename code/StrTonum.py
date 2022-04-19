import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
path ='D:/University\ML\dataset/UNSW_NB15_training-set.csv'
data_set = pd.read_csv(path)

proto = data_set[['proto']]
# print(proto.shape)
le = LabelEncoder()
le.fit(proto)
print('class:', le.classes_)
print((le.classes_).shape)
proto_process = pd.DataFrame(le.transform(proto))
data_set[['proto']] = proto_process

# label11.to_csv('D:/University\ML\dataset/1.csv',header=None)
# label11 = pd.DataFrame(min_max_scaler.fit_transform(label11))
# label11.to_csv('D:/University\ML\dataset/2.csv',header=None)
# print('encode class', le.transform(label1))
# print('min_max_scaler', label11)

service = data_set[['service']]
le.fit(service)
print('class:', le.classes_)
service_process = pd.DataFrame(le.transform(service))
data_set[['service']] = service_process
print(service_process)

state = data_set[['state']]
le.fit(state)
print('class:', le.classes_)
state_process = pd.DataFrame(le.transform(state))
data_set[['state']] = state_process
print(state_process)

attack_cat = data_set[['attack_cat']]
le.fit(attack_cat)
print('class', le.classes_)
attack_cat_process = pd.DataFrame(le.transform(attack_cat))
data_set[['attack_cat']] = attack_cat_process
print(attack_cat_process)

data_set.to_csv('D:/University\ML\dataset/transform_process.csv', index=False)

