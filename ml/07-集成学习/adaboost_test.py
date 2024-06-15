import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

np.set_printoptions(suppress=True)

data = pd.read_csv('./data/risk_factors_cervical_cancer.csv')
data.replace('?', np.nan, inplace=True)
knn = KNNImputer()
data = pd.DataFrame(knn.fit_transform(data))

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
Y.replace(0.0, -1.0, inplace=True)
# Y = np.array(Y)
# print(list(Y).count(1))

# X = np.array(X)

# 上采样
smote = SMOTE()
X, Y = smote.fit_resample(X=X, y=Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)


class Adaboost:
    def __init__(self, x, y, n_estimators=50, learning_rate=1., max_depth=2, tolerance=10e-5):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.w = np.ones(x.shape[0]) / x.shape[0]
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.models = []
        self.alphas = []

    def fit(self):
        for i in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(self.x, self.y, sample_weight=self.w)
            self.models.append(model)

            # 计算误差
            e = sum(self.w[self.y != model.predict(self.x)])
            print(f'第{i}轮误差：{e}')

            if e <= self.tolerance:
                self.alphas.append(1)
                return

            # 计算学习器权重
            alpha = 0.5 * np.log2((1-e) / e)
            self.alphas.append(alpha)
            # 更新权值
            y_hat = alpha * model.predict(self.x)
            result = - self.y * y_hat
            z = np.sum(self.w * np.power(2, result))
            self.w = (self.w / z) * np.power(2, result)

    def predict(self, x):
        y_hat = np.zeros(x.shape[0])
        for i in range(len(self.models)):
            y_hat += self.learning_rate * self.alphas[i] * np.array(self.models[i].predict(x))

        return np.sign(y_hat)


# ------------------------------------------------------------------------------------------------
df = pd.DataFrame([[0, 1],
                   [1, 1],
                   [2, 1],
                   [3, -1],
                   [4, -1],
                   [5, -1],
                   [6, 1],
                   [7, 1],
                   [8, 1],
                   [9, -1]])

X1 = df.iloc[:, :-1]
Y1 = df.iloc[:, -1]
# -------------------------------------------------------------------------------------------------


data = pd.read_csv('./data/iris.data', header=None)
data = data.iloc[50:, :]
X = data.iloc[:, 0:4]
Y = data.iloc[:, 4]
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
# -------------------------------------------------------------------------------------------------

adaboost = Adaboost(x_train, y_train, n_estimators=10, max_depth=1, learning_rate=0.1)
adaboost.fit()
y_train_hat = adaboost.predict(x_train)
# print(y_train_hat)

y_test_hat = adaboost.predict(x_test)
# print(y_test_hat)

print("-" * 25)
print(f'训练集得分：{f1_score(y_train_hat, y_train, average=None)}')
print(f'测试集得分：{f1_score(y_test_hat, y_test, average=None)}')

print('-' * 25)
ada = AdaBoostClassifier(n_estimators=15, learning_rate=0.1)
ada.fit(x_train, y_train)
print(f'sklearn模型训练集得分：{ada.score(x_train, y_train)}')
print(f'sklearn模型测试集得分：{ada.score(x_test, y_test)}')
