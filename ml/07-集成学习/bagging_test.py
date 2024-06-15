import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys


iris_lst = datasets.load_iris()
# print(iris_lst.data)
# print(iris_lst.target)

X = pd.DataFrame(iris_lst.data)
Y = pd.DataFrame(iris_lst.target)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

data_train = x_train.assign(target=y_train)
# sys.exit(0)

models = []
n_trees = 100

for n in range(n_trees):
    data = data_train.sample(frac=1.0, replace=True)
    x = data.iloc[:, 0:4]
    y = data.iloc[:, 4]
    ctree = DecisionTreeClassifier(max_depth=1)
    ctree.fit(x, y)
    models.append(ctree)

y_train_hats = []
y_test_hats = []
for model in models:
    y_train_hats.append(list(model.predict(x_train)))
    y_test_hats.append(list(model.predict(x_test)))

y_train_hats = np.array(y_train_hats).T.tolist()
y_test_hats = np.array(y_test_hats).T.tolist()

y_train_hat = []
y_test_hat = []
for lst in y_train_hats:
    max_v = max(lt := [lst.count(i) for i in range(3)])
    y_train_hat.append(lt.index(max_v))

for lst in y_test_hats:
    max_v = max(lt := [lst.count(i) for i in range(3)])
    y_test_hat.append(lt.index(max_v))

# print(y_train_hat)
# print(y_test_hat)

# train_score = f1_score(y_train_hat, y_train, average='micro')
train_score = f1_score(y_train_hat, y_train, average=None)
# test_score = f1_score(y_test_hat, y_test, average='micro')
test_score = f1_score(y_test_hat, y_test, average=None)

print(f'训练集得分：{train_score}')
print(f'测试集得分：{test_score}')
