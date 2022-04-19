# 机器学习及深度学习实现IDS✨

## 1、说明

- 本文档用来记录机器学习课程作业，完成机器学习及深度学习实现入侵检测。
- 数据集为UNSW_NB15官方数据集，42列特征，一列标签，二分类问题；

- 采用的方法为：① 特征处理；② Pearson过滤和随机森林特征选择；③ CNN训练

- 参考的文档为《基于深度神经网络的网络入侵检测技术》

[数据集官网:](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

[数据集下载地址](https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSW-NB15%20-%20CSV%20Files)

## 2、代码

运行代码前确保安装有sklearn包和Tensorflow以及keras，注意Tensorflow和keras的对应关系。

`pip install -r requirements.txt`安装环境依赖即可

代码的结构如下：

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220419092935.png" style="zoom: 67%;" />

`StrTonum.py`：将数值型特征转化为字符型，才能送入模型训练；

`Min-max`：对数据进行归一化处理，加快模型的训练；

`Balanced.py`：对正负样本个数进行平衡处理，参考论文中ADASYN算法；

`Pearson.py`：计算不同特征及之间的相关性系数；

`Find_feature.py`：随机森林进行特征选择；

`Pro_Testdata.py`：对测试数据进行处理，确保测试集和在训练集上筛选出的特征一致；

`CNN.py`：keras搭建CNN网络进行训练；

整个代码的运行流程按上述流程即可;

## 3、CNN的结构

CNN为自己搭建的网络，网络的结构如下：

两层conv，两层pool、两层全连接，以及Dropout。

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220419093337.png" style="zoom: 67%;" />

## 4、效果

通过特征选择以及平衡正负样本个数，最终CNN在测试集上的准确率为89%

`train`：

![](https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220419093554.png)

`test`：

![](https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220419093937.png)

## 5、Later

未完待续！

