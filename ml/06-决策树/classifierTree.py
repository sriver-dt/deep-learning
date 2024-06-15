import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn import tree

data = pd.read_csv('./data/iris.data', header=None)

X = data.iloc[:, 0:4]
Y = data.iloc[:, 4]

encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
labels = encoder.classes_

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

ctree = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=5, min_samples_leaf=3)

ctree.fit(x_train, y_train)
train_score = ctree.score(x_train, y_train)
test_score = ctree.score(x_test, y_test)
print(f'训练集得分：{train_score}')
print(f'测试集得分：{test_score}')

y_hat = ctree.predict(X)
print(y_hat)

plt.plot(range(len(Y)), Y, 'go', markersize=10, zorder=5, label='train')
plt.plot(range(len(Y)), y_hat, 'ro', markersize=5, zorder=10, label='test')
plt.show()


# 可视化
dot_data = tree.export_graphviz(decision_tree=ctree, out_file=None,
                                feature_names=['A', 'B', 'C', 'D'],
                                class_names=[labels[0], labels[1], labels[2]],
                                filled=True, rounded=True,
                                special_characters=True,
                                node_ids=True
                                )

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("irisTree.png")
graph.write_pdf("irisTree.pdf")
