import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris


data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=5)

train_data = lgb.Dataset(x_train, label=y_train)
validation_data = lgb.Dataset(x_test, label=y_test)

params = {
    'learning_rate': 0.1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'max_depth': 4,
    'objective': 'multiclass',  # 目标函数
    'num_class': 3,
    'verbose': -1,
}

lgbm = lgb.train(params, train_data, valid_sets=[validation_data])

y_train_hat = lgbm.predict(x_train)
y_train_hat = [list(x).index(max(x)) for x in y_train_hat]
y_test_hat = lgbm.predict(x_test)
y_test_hat = [list(x).index(max(x)) for x in y_test_hat]
print(f'训练集得分：{f1_score(y_train_hat, y_train, average=None)}')
print(f'测试集得分：{f1_score(y_test_hat, y_test, average=None)}')
