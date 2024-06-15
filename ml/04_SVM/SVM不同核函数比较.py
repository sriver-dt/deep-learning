import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


plt.rcParams['font.sans-serif'] = ['SimHei']

data = pd.read_csv('./data/iris.data', header=None)
# print(data)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
# print(X.info())
Y.replace('Iris-setosa', 0, inplace=True)
Y.replace('Iris-versicolor', 1, inplace=True)
Y.replace('Iris-virginica', 2, inplace=True)
# print(Y.tolist())

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=5)

linear = SVC(kernel='linear', C=1)
rbf = SVC(kernel='rbf', C=1, gamma=0.1)
poly = SVC(kernel='poly', C=1, gamma=0.1, coef0=1, degree=2)
sigmoid = SVC(kernel='sigmoid', C=1, gamma=0.01, coef0=0.1)

train_scores = []
test_scores = []
times = []
models = {'linear': linear, 'rbf': rbf, 'poly': poly, 'sigmoid': sigmoid}
for model in models.values():
    start = time.perf_counter()
    model.fit(x_train, y_train)
    end = time.perf_counter()
    times.append(end-start)
    train_scores.append(model.score(x_train, y_train))
    test_scores.append(model.score(x_test, y_test))

print('训练时间：', times)
print('训练集得分：', train_scores)
print('测试集得分：', test_scores)

plt.subplot(1, 2, 1)
plt.plot(models.keys(), times, c='b', label="time")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(models.keys(), train_scores, c='r', label="train")
plt.plot(models.keys(), test_scores, c='g', label="test")
plt.legend()
plt.show()
