import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 防止画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
# 不显示科学计数法
np.set_printoptions(suppress=True)


data = pd.read_csv('./data/breast-cancer-wisconsin.data', sep=',', header=None)

# 数据清洗
data.replace('?', np.nan, inplace=True)
data = data.dropna(axis=0)
data = data.astype(np.int64)
# print(data.info())

X = data.iloc[:, 1:-1]
Y = data.iloc[:, -1]
# print(X.info())

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

logistic = LogisticRegression()
logistic.fit(x_train, y_train)
print('训练集得分：', logistic.score(x_train, y_train))
print('测试集得分：', logistic.score(x_test, y_test))
y_hat = logistic.predict(x_test)

plt.plot(range(len(y_hat)), y_test, 'ro', markersize=10, zorder=5, label='标签值')
plt.plot(range(len(y_hat)), y_hat, 'go', markersize=5, zorder=10, label='预测值')
plt.show()
