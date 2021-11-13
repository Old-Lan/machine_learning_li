import numpy as np
import pandas as pd
#实现思路
#1.初始化w、b.
#2.遍历数据集，如果y(w*x+b)<=0,则更新w、b
#3.得到新的模型，回到步骤2.


## 读取数据
def read_data(path):
    data = pd.read_csv(path,header=None)
    # print(data)
    X = data.iloc[:,:2].values
    # print(X)
    # print(type(X))
    y = data.iloc[:,2].values
    # print(y)
    # print(type(y))
    return X,y

## 数据分布
import matplotlib.pyplot as plt
def data_show(X,w=None):
    if X is not None:
        plt.scatter(X[:50,0],X[:50,1], color='red', marker='o', label='+1')
        plt.scatter(X[50:,0],X[50:,1], color='blue', marker='x', label='-1')
        plt.xlabel('Feature1')
        plt.ylabel('Feature2')
        plt.title('original')
    if w is not None:
        x1 = -2
        x2 = 2
        y1 = -(w[0]+w[1]*x1)/w[2]
        y2 = -(w[0]+w[1]*x2)/w[2]
        plt.plot([x1,x2],[y1,y2],'r')
    plt.show()

## 特征归一化
def feature_normalization(X):
    u = np.mean(X,axis=0)
    o = np.std(X,axis=0)
    # print(u)
    # print(o)
    X = (X-u)/o
    # print(X)
    return X

## 初始化超平面
def init_hyperplane(X):
    X = np.hstack((np.ones((X.shape[0],1)),X))
    w = np.random.randn(3,1)
    return X,w

## 训练
def train(X,y):
    ### 初始化参数
    X,w = init_hyperplane(X)
    for i in range(X.shape[0]):
        score = np.dot(X,w)
        y_train = np.ones_like(y)
        # print(y_train)
        positions_neg = np.where(score < 0)[0]
        y_train[positions_neg] = -1
        count_error = len(np.where(y!=y_train)[0])
        if count_error > 0:
            t = np.where(y!=y_train)[0][0]
            w += y[t]*X[t,:].reshape((3,1))
    return w





if __name__ == '__main__':
    X,y = read_data('./data/data1_pla.csv')## 读取数据
    # print(X)
    X = feature_normalization(X)## 特征归一化处理
    data_show(X)
    w = train(X,y) ##训练数据
    data_show(X,w)