import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


data = pd.read_csv('./data/iris.data', header=None)

X = data.iloc[:, 0:4]
Y = data.iloc[:, 4]
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
# print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

gauss = GaussianNB()
bernoulli = BernoulliNB()
mult = MultinomialNB()

models = [gauss, bernoulli, mult]

train_score = []
test_score = []
for model in models:
    model.fit(x_train, y_train)
    train_score.append(model.score(x_train, y_train))
    test_score.append(model.score(x_test, y_test))

print(f'训练集得分：{train_score}')
print(f'测试集得分：{test_score}')
