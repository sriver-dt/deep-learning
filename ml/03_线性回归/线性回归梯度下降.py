import matplotlib.pyplot as plt
import time


plt.rcParams['font.sans-serif'] = ['SimHei']    # 防止画图中文乱码


class MyLinearMod:

    def __init__(self):
        self.theta = [0, 0, 0]
        self.R2_score = -1

    @staticmethod
    def polynomial(lst, degree=2, not_intercept=False):
        """
            简单的单个特征的多项式扩展
            not_intercept: 默认为False表示需要添加截距项
        """
        new_lst = []
        for i in lst:
            temp = [i ** j for j in range(not_intercept, degree + 1)]
            new_lst.append(temp)
        return new_lst

    @staticmethod
    def min_max(x_data, y_data, has_intercept=True):
        """ 对x, y 分别进行 Min-Max归一化 """
        y_max = max(y_data)
        y_min = min(y_data)
        y_data = [(v - y_min) / (y_max - y_min) for v in y_data]

        # 如果有截距项，截距项不需要做Min-Max归一化
        if has_intercept:
            for n in range(1, len(x_data[0])):
                x_n_max = max(lst := [v[n] for v in x_data])   # 数据样本每个特征的最大值
                x_n_min = min(lst)                             # 数据样本每个特征的最小值
                for i in range(len(x_data)):
                    x_data[i][n] = (x_data[i][n] - x_n_min) / (x_n_max - x_n_min)
        else:
            for n in range(len(x_data[0])):
                x_n_max = max(lst := [v[n] for v in x_data])   # 数据样本每个特征的最大值
                x_n_min = min(lst)                             # 数据样本每个特征的最小值
                for i in range(len(x_data)):
                    x_data[i][n] = (x_data[i][n] - x_n_min) / (x_n_max - x_n_min)
        return x_data, y_data

    def r2_score(self, x_t, y_t):
        """ 计算 r2 得分，无返回值，直接修改实例的 R2_score """
        size = len(x_t)
        rss, tss = 0, 0
        y_mean = sum(y_t) / size    # Y的平均值
        for i in range(size):
            y_hat = x_t[i][0] * self.theta[0] + x_t[i][1] * self.theta[1] + x_t[i][2] * self.theta[2]
            rss += (y_t[i] - y_hat) ** 2

        tss = sum([(v - y_mean)**2 for v in y_t])
        self.R2_score = 1 - rss / tss

    def costs(self, x_t, y_t):
        """ 求损失 """
        size = len(x_t)
        cost = 0
        for i in range(size):
            y_hat = x_t[i][0] * self.theta[0] + x_t[i][1] * self.theta[1] + x_t[i][2] * self.theta[2]
            cost += (y_hat - y_t[i]) ** 2

        cost = 1 / (2 * size) * cost
        return cost

    def y_predict(self, x_set):
        """ 预测值 """
        size = len(x_set)
        y_hat_lst = []
        for i in range(size):
            y_hat = x_set[i][0] * self.theta[0] + x_set[i][1] * self.theta[1] + x_set[i][2] * self.theta[2]
            y_hat_lst.append(y_hat)
        return y_hat_lst

    def gradient_descent_bgd(self, x_t, y_t, theta, alpha=0.01, max_iter=5000):
        """ 批量梯度下降, 同时返回下降过程的损失列表 """
        cost_lst = []
        self.theta = theta
        a = alpha * (1 / len(x_t))

        for i in range(max_iter):
            gradient_sum0 = gradient_sum1 = gradient_sum2 = 0
            cost = 0
            for j in range(len(x_t)):
                y_hat = x_t[j][0] * self.theta[0] + x_t[j][1] * self.theta[1] + x_t[j][2] * self.theta[2]
                gradient_sum0 += (y_hat - y_t[j]) * x_t[j][0]
                gradient_sum1 += (y_hat - y_t[j]) * x_t[j][1]
                gradient_sum2 += (y_hat - y_t[j]) * x_t[j][2]

                cost += (y_hat - y_t[j]) ** 2

            # 更新梯度
            self.theta[0] -= a * gradient_sum0
            self.theta[1] -= a * gradient_sum1
            self.theta[2] -= a * gradient_sum2
            # 计算损失
            cost = 1 / (2 * len(x_t)) * cost
            cost_lst.append(cost)

        return cost_lst


x = [290., 329., 342., 359., 369., 386., 395., 410., 425., 427., 433., 437., 445., 450., 458., 462., 469., 478., 484.,
     489., 495., 496., 502., 509., 511., 514., 516., 518., 521., 523.]
y = [36302., 15125., 10094., 5045., 2885., 590., 77., 302., 1877., 2189., 3269., 4109., 6077, 7502., 10094., 11534.,
     14285., 18254., 21170., 23765., 27077., 27650., 31214., 35645., 36965., 38990., 40370., 41774., 43925., 45389.]
# plt.scatter(x, y)
# plt.xlim(0, 50000)
# plt.ylim(0, 50000)
# plt.show()

x1 = MyLinearMod.polynomial(x, degree=2)    # 扩展二次多项式
X, Y = MyLinearMod.min_max(x1, y)           # Min-Max归一化
# X, Y = x1, y

linear = MyLinearMod()
start_time = time.time()
costs_lst = linear.gradient_descent_bgd(X, Y, theta=[0, 0, 0], alpha=1, max_iter=85000)   # 批量梯度下降
end_time = time.time()
linear.r2_score(X, Y)
print(f'训练时间: {end_time - start_time}')
print(f'theta: {linear.theta}')
print(f'R2_score: {linear.R2_score}')

# -----------画图------------
plt.figure()
plt.subplot(1, 2, 1)
y_cost = costs_lst[:]
plt.plot(range(len(y_cost)), y_cost, label='损失函数')
plt.legend()

plt.subplot(1, 2, 2)
y_hats = linear.y_predict(X)
x1 = [i[1] for i in X]
plt.plot(x1, y_hats, c='g', label='预测值')
plt.scatter(x1, Y, c='r', label='真值')
plt.legend()
plt.show()
