import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import time


np.set_printoptions(suppress=True)

data = pd.read_csv('./data/risk_factors_cervical_cancer.csv')

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# 查看每列有几种值
# print(X.nunique())

# 查看每列有哪几种值
# print(a := X.apply(lambda x: x.unique()))

# 数据清洗
X.replace('?', np.nan, inplace=True)
# 缺省值填充
simple_impute = SimpleImputer(strategy='most_frequent')
X1 = simple_impute.fit_transform(X)
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, Y, test_size=0.3, random_state=5)

knn_impute = KNNImputer(n_neighbors=5)
X2 = knn_impute.fit_transform(X)
x2_train, x2_test, y2_train, y2_test = train_test_split(X2, Y, test_size=0.3, random_state=5)


models = [Pipeline([('PCA', PCA()),
                    ('random_forest', RandomForestClassifier())
                    ]),
          Pipeline([('LDA', LatentDirichletAllocation()),
                    ('random_forest', RandomForestClassifier())
                    ])
          ]

# 网格调参
params1 = {'PCA__n_components': [0.2, 0.4, 0.5, 0.7, 0.9],
           'random_forest__n_estimators': [20, 50, 80, 100],
           'random_forest__max_depth': [5, 10, 15, 20]
           }

params2 = {'LDA__n_components': [5, 8, 10],
           'random_forest__n_estimators': [20, 50, 80, 100],
           'random_forest__max_depth': [5, 10, 15, 20]
           }

model1 = GridSearchCV(models[0], param_grid=params1, cv=5)
model2 = GridSearchCV(models[1], param_grid=params2, cv=5)

start = time.time()
model1.fit(x1_train, y1_train)
end = time.time()
print(f'model1最优参数：{model1.best_params_}')
print(f'运行时间：{end - start}')

start = time.time()
model2.fit(x2_train, y2_train)
end = time.time()
print(f'model2最优参数：{model2.best_params_}')
print(f'运行时间：{end - start}')
print('-' * 25)
print(f'SimpleImputes 的训练集得分：{model1.score(x1_train, y1_train)}')
print(f'SimpleImputes 的测试集得分：{model1.score(x1_test, y1_test)}')
print('-' * 25)
print(f'KNNImputes 的训练集得分：{model2.score(x2_train, y2_train)}')
print(f'KNNImputes 的测试集得分：{model2.score(x2_test, y2_test)}')
