import joblib
import numpy as np
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay


# 设置字体支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# 下载完整数据
fetch_20newsgroups(data_home='./data/total', subset='all')
# 下载remove 'headers', 'footers', 'quotes' 的数据
fetch_20newsgroups(data_home='./data/remove', subset='all', remove=('headers', 'footers', 'quotes'))

datas = joblib.load('./data/total/20news-bydate_py3.pkz')


def data_show(data):
    print(type(data))
    print(data.keys())
    print(data['train'].keys())
    print('-' * 25)
    print(type(data['train']['data'][0]))
    print('-' * 25)
    train_data = data['train']
    print(len(train_data['data']))
    print('- * 25')
    print(train_data['filenames'])
    print('- * 25')
    print(len(train_data['target_names']))
    print('- * 25')
    print(len(train_data['target']))
    print('- * 25')
    print(train_data['DESCR'])


def save_data(data, filename):
    data.to_csv(filename, index=False)


# 文本向量化
def text_vector(train_data, test_data):
    tf_idf = TfidfVectorizer(max_df=0.5, min_df=5, stop_words='english')
    tf_idf.fit(train_data)
    x_train = tf_idf.transform(train_data)
    x_test = tf_idf.transform(test_data)
    return x_train, x_test


# 通过网格调参选取最优超参数模型
def model_train(x, y, model_, params):
    grid_model = GridSearchCV(model_, param_grid=params)
    s = time.time()
    grid_model.fit(x, y)
    e = time.time()
    return e-s, grid_model, grid_model.best_estimator_, grid_model.best_score_


# 初始化多个模型和超参数列表
def models():
    model_lst = []

    # LogisticRegression
    logistic = LogisticRegression()
    # logistic_params = {'C': [3, 5, 10],
    #                    'max_iter': [200, 250, 300]
    #                    }
    logistic_params = {'C': [10],
                       'max_iter': [200]
                       }
    # best_estimator: LogisticRegression(C=10, max_iter=200)
    model_lst.append(((logistic, "LogisticRegression"), logistic_params))

    # RandomForestClassifier
    random_forest = RandomForestClassifier()
    # rf_params = {'n_estimators': [50, 100, 150],
    #              'max_depth': [30, 50, 80, 100]
    #              }
    rf_params = {'n_estimators': [100],
                 'max_depth': [100]
                 }
    # best_estimator: RandomForestClassifier(max_depth=100, n_estimators=100)
    model_lst.append(((random_forest, "RandomForestClassifier"), rf_params))

    # # AdaBoostClassifier
    # adaboost = AdaBoostClassifier()
    # ada_params = {'n_estimators': [150],
    #               'learning_rate': [0.5],
    #               }
    # model_lst.append(((adaboost, "AdaBoostClassifier"), ada_params))

    # LinearSVC
    svc = LinearSVC(dual=False)
    # svc_params = {'C': [0.5, 1, 2],
    #               'max_iter': [50, 100, 150],
    #               }
    svc_params = {'C': [1],
                  'max_iter': [50],
                  }
    # best_estimator: LinearSVC(C=1, dual=False, max_iter=50)
    model_lst.append(((svc, "LinearSVC"), svc_params))

    # ComplementBN
    naive_bayes = ComplementNB()
    # nb_params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
    nb_params = {'alpha': [1.0]}
    # best_estimator: ComplementNB(alpha=1.0)
    model_lst.append(((naive_bayes, "ComplementBN"), nb_params))

    # KNN
    knn = KNeighborsClassifier()
    # knn_params = {'n_neighbors': [3, 5, 10]}
    knn_params = {'n_neighbors': [5]}
    # best_params: {'n_neighbors': 5}
    model_lst.append(((knn, "KNN"), knn_params))

    return model_lst


def train(x, y):
    train_times_ = []
    scores_ = []
    models_ = []
    i = 1
    for mod, params in models():
        print(f'第 {i} 次, {mod[1]}开始训练 -----------------------------')
        train_time, model_, best_estimator, best_score = model_train(x, y, mod[0], params)
        print(f'训练时间: {train_time}')
        print(f'best_estimator: {best_estimator}')
        print(f'best_params: {model_.best_params_}')
        print(f'best_score: {best_score}')
        s = time.time()
        y_hat = model_.predict(X_test)
        e = time.time()
        predict_time = e - s
        print(f'预测时间: {predict_time}')
        score = f1_score(y_hat, y_test, average='macro')
        print(f'测试集得分: {score}')

        train_times_.append(train_time)
        scores_.append(score)
        models_.append((model_, mod[1]))
        i += 1

    return train_times_, scores_, models_


def save_model(model_, name):
    joblib.dump(model_, './model/'+name+'.model')


def load_model(model_name):
    return joblib.load('./model/'+model_name+'.model')


def sava_params(tm, f1, modes):
    with open('./model/models_params.txt', mode='w') as f:
        for i in range(len(modes)):
            f.write(f"模型名称：{modes[i][1]} ")
            f.write(f"模型训练时间：{tm[i]} ")
            f.write(f"模型f1_score：{f1[i]} ")
            f.write("\n")


def load_params():
    with open('./model/models_params.txt', mode='r') as file:
        parma_str = file.read()

        p = re.compile(r'模型名称：(\w+)\s')
        model_names_ = p.findall(parma_str)

        p = re.compile(r'模型训练时间：([\d.]+)\s')
        train_times_ = p.findall(parma_str)
        train_times_ = [float(s) for s in train_times_]

        p = re.compile(r'模型f1_score：([\d.]+)\s')
        scores_ = p.findall(parma_str)
        scores_ = [float(s) for s in scores_]
    return model_names_, train_times_, scores_


def roc_pr(x, y, model_):
    y_pred_proba = model_.predict_proba(x)[:1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    roc_display.plot()
    plt.title("ROC Curve")

    plt.subplot(122)
    pr_display.plot()
    plt.title("Precision-Recall Curve")

    plt.show()


def draw(tm, f1, m):
    plt.figure(figsize=(10, 5))
    plt.scatter(tm, f1, color='red', s=25)
    for i in range(len(m)):
        plt.annotate(m[i], xy=(tm[i], f1[i]), xytext=(tm[i]-10, f1[i]+0.003))
    plt.title("训练时间与测试得分")
    plt.xlabel("训练时间(s)")
    plt.ylabel("f1_score")
    plt.show()


if __name__ == '__main__':
    # data_show(datas)
    data_train = pd.DataFrame({'data': datas['train']['data'], 'label': datas['train']['target']})
    data_test = pd.DataFrame({'data': datas['test']['data'], 'label': datas['test']['target']})
    save_data(data_train, './data/train_data.data')
    save_data(data_test, './data/test_data.data')

    # 加载数据集
    train_datas = pd.read_csv('./data/train_data.data')
    test_datas = pd.read_csv('./data/test_data.data')
    # print(train_datas.head(5))
    # print(train_datas.dtypes)
    # print(train_datas.info())

    # 提取标签数据
    y_train = train_datas['label']
    y_test = test_datas['label']

    # X TF-IDF向量化
    start = time.time()
    X_train, X_test = text_vector(train_datas['data'], test_datas['data'])
    end = time.time()
    print(f'TF-IDF 向量化耗时：{end - start}')
    print('')
    print(f'X_train 的类型：{type(X_train)}')   # 稀疏矩阵
    print(f'X_train 的形状：{X_train.shape}')
    print(f'X_test 的形状：{X_test.shape}')

    # 训练模型
    train_times, scores, mods = train(X_train, y_train)

    # 存储模型
    for m, n in mods:
        save_model(m, n)
    # 存储各种参数
    sava_params(train_times, scores, mods)

    # 加载参数
    model_names, train_times, scores = load_params()
    # 可视化模型性能比较
    draw(train_times, scores, model_names)

    # # 加载模型
    # for mod_name in model_names:
    #     model = load_model(mod_name)
    #     # roc 仅支持二分类, 需要对数据集做处理
    #     # roc_pr(X_test, y_test, model)
