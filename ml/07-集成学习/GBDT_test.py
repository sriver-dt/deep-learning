import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


data = pd.read_csv('./data/boston_housing.data', sep='\s+', header=None)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

gbdt = GradientBoostingRegressor(n_estimators=40, loss='squared_error', learning_rate=0.1, subsample=0.5)

gbdt.fit(x_train, y_train)

print(f'训练集得分：{gbdt.score(x_train, y_train)}')
print(f'测试集得分：{gbdt.score(x_test, y_test)}')
