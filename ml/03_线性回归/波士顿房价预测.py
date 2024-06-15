import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')   # 忽略警告
np.set_printoptions(suppress=True)
plt.rcParams['font.sans-serif'] = ['SimHei']    # 防止画图中文乱码

# 读取数据
data = pd.read_csv('./data/boston_housing.data', sep='\s+', header=None)
# print(data)

# 划分特征和标签
x = data.iloc[:, :-1]
# X = np.array(x)
y = data.iloc[:, -1]
# Y = np.array(y)

# 划分训练集、测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)
# print(np.info(x_train))
# # 特征多项式扩展
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# 构建模型
# 03_线性回归
linear = LinearRegression()
linear.fit(x_train_poly, y_train)
y_test_hat = linear.predict(x_test_poly)
print('linearRegression', '-'*50)
# print(linear.coef_)
# print(linear.intercept_)
print(linear.score(x_train_poly, y_train))
print(linear.score(x_test_poly, y_test))
plt.plot(range(len(y_test)), y_test, c='r')
plt.scatter(range(len(y_test)), y_test_hat, c='g')
plt.show()

# ElasticNet
elastic = ElasticNetCV(alphas=[0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001], l1_ratio=[0, 1, 0.1, 10, 0.001, 0.01])
elastic.fit(x_train_poly, y_train)
y_test_hat = elastic.predict(x_test_poly)
print('ElasticNet', '-'*50)
# print(elastic.coef_)
# print(elastic.intercept_)
print(elastic.score(x_train_poly, y_train))
print(elastic.score(x_test_poly, y_test))
print(elastic.alpha_)
print(elastic.l1_ratio_)
plt.plot(range(len(y_test)), y_test, c='r')
plt.scatter(range(len(y_test)), y_test_hat, c='g')
plt.show()
