import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer


data_r = pd.read_csv('./data/boston_housing.data', sep='\s+', header=None)

X_r = data_r.iloc[:, :-1]
Y_r = data_r.iloc[:, -1]

x_r_train, x_r_test, y_r_train, y_r_test = train_test_split(X_r, Y_r, test_size=0.3, random_state=5)

xgbr = XGBRegressor(booster='gbtree', eta=0.1)
xgbr.fit(x_r_train, y_r_train)
print(f'训练集得分：{xgbr.score(x_r_train, y_r_train)}')
print(f'测试集得分：{xgbr.score(x_r_test, y_r_test)}')
print('-' * 25)


data_c = pd.read_csv('./data/risk_factors_cervical_cancer.csv')
data_c.replace('?', np.nan, inplace=True)

X_c = data_c.iloc[:, :-1]
Y_c = data_c.iloc[:, -1]
simple_impute = SimpleImputer(strategy='most_frequent')
X_c = simple_impute.fit_transform(X_c)

x_c_train, x_c_test, y_c_train, y_c_test = train_test_split(X_c, Y_c, test_size=0.3, random_state=5)

xgbc = XGBClassifier(eta=0.1)
xgbc.fit(x_c_train, y_c_train)
print(f'训练集得分：{xgbc.score(x_c_train, y_c_train)}')
print(f'测试集得分：{xgbc.score(x_c_test, y_c_test)}')
