import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution1D,MaxPooling1D,Flatten
from tensorflow.keras.optimizers import Adam
train_path = 'D:/University/ML/dataset/train_s_f.csv'
test_path = 'D:/University/ML/dataset/test_s_f.csv'
trainset = pd.read_csv(train_path)
testset = pd.read_csv(test_path)
x_train = trainset.iloc[:,0:20]
y_train = trainset.iloc[:,-1]
x_test = testset.iloc[:,0:20]
y_test = testset.iloc[:,-1]
x_train = x_train.values.reshape(-1, 20, 1)
x_test = x_test.values.reshape(-1, 20, 1)
# 把训练集和测试集的标签转为独热编码
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
# 定义顺序模型
model = Sequential([

])
# 第一个卷积层
# input_shape 输入数据
# filters 滤波器个数32，生成32 张特征图
# kernel_size 卷积窗口大小3*3
# strides 步长1
# padding padding方式 same/valid
# activation 激活函数
model.add(Convolution1D(
                         input_shape=(20, 1),
                         filters=32,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         activation='relu'
                        ))
# 第一个池化层
# pool_size 池化窗口大小2*2
# strides 步长2
# padding padding方式 same/valid
model.add(MaxPooling1D(
                        pool_size=2,
                        strides=2,
                        padding='same',
                      ))
# 第二个卷积层
# filters 滤波器个数64，生成64 张特征图
# kernel_size 卷积窗口大小3*3
# strides 步长1
# padding padding方式 same/valid
# activation 激活函数
model.add(Convolution1D(64, 3, strides=1, padding='same', activation='relu'))
# 第二个池化层
# pool_size 池化窗口大小2*2
# strides 步长2
# padding padding方式 same/valid
model.add(MaxPooling1D(2, 2, 'same'))
# 把第二个池化层的输出进行数据扁平化
model.add(Flatten())
# 第一个全连接层
model.add(Dense(1024, activation='relu'))
# Dropout
model.add(Dropout(0.5))
# 第二个全连接层
model.add(Dense(2, activation='softmax'))
# 定义优化器
adam = Adam(lr=1e-4)
# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
# 保存模型
model.save('model.h5')
loss_acc = model.evaluate(x_test,y_test)
print('On Testset, loss={}, acc={}'.format(loss_acc[0], loss_acc[1]))