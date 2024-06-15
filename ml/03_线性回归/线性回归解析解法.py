import matplotlib.pyplot as plt
import time


# 多项式扩展
def polynomial(lst, degree=2, not_intercept=False):
    new_lst = []
    for i in lst:
        temp = [i ** j for j in range(not_intercept, degree + 1)]
        new_lst.append(temp)
    return new_lst


# 解析解法
def resolve(x_p, y):
    size = len(x_p[0])
    new_x = []
    # 先求 x 的转置和 x 的点乘
    for i in range(size):
        temp = []
        for j in range(size):
            s = 0
            for k in range(len(x_p)):
                s += x_p[k][i] * x_p[k][j]
            temp.append(s)
        new_x.append(temp)

    # 计算行列式
    x_det = det(new_x)
    if x_det == 0:
        # 如果 x 的行列式为 0，则不可逆,加一个扰动
        x_det = 1
        for i in range(size):
            x_det *= new_x[i][i] * 0.0001

    # 求伴随矩阵
    x_a = adjoint(new_x)

    # 求矩阵的逆
    for i in range(size):
        for j in range(size):
            x_a[i][j] = x_a[i][j] / x_det

    # 中间计算过程
    temp = []
    for i in range(size):
        rows = []
        for k in range(len(x)):
            s = x_a[i][0] * x_p[k][0] + x_a[i][1] * x_p[k][1] + x_a[i][2] * x_p[k][2]
            rows.append(s)
        temp.append(rows)

    # 计算theta
    theta = []
    for m in range(size):
        s = 0
        for n in range(len(x_p)):
            s += temp[m][n] * y[n]
        theta.append(s)
    return theta


# 余子式
def cofactor(A, m, n):
    cof = []
    for i in range(len(A)):
        if i != m:
            row = []
            for j in range(len(A)):
                if j != n:
                    row.append(A[i][j])
            cof.append(row)
    return cof


# 行列式计算
def det(A):
    size = len(A)
    if size == 1:
        return A[0][0]
    if size == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    result = 0
    for i in range(size):
        result += A[0][i] * det(cofactor(A, 0, i)) * (-1) ** i
    return result


# 伴随矩阵
def adjoint(A):
    size = len(A)
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            s = ((-1) ** (i + j + 2)) * det(cofactor(A, i, j))
            row.append(s)
        result.append(row)
    return result


# 计算 R2_score
def r2_score(theta, x_t, y_t):
    size = len(x_t)
    rss, tss = 0, 0
    y_mean = sum(y_t) / size    # Y的平均值
    for i in range(size):
        y_hat = x_t[i][0] * theta[0] + x_t[i][1] * theta[1] + x_t[i][2] * theta[2]
        rss += (y_t[i] - y_hat) ** 2

    tss = sum([(v - y_mean)**2 for v in y_t])
    return 1 - rss / tss


x = [290., 329., 342., 359., 369., 386., 395., 410., 425., 427., 433., 437., 445., 450., 458., 462., 469., 478., 484.,
     489., 495., 496., 502., 509., 511., 514., 516., 518., 521., 523.]
y = [36302., 15125., 10094., 5045., 2885., 590., 77., 302., 1877., 2189., 3269., 4109., 6077, 7502., 10094., 11534.,
     14285., 18254., 21170., 23765., 27077., 27650., 31214., 35645., 36965., 38990., 40370., 41774., 43925., 45389.]

x_ploy = polynomial(x)
start = time.perf_counter()
theta = resolve(x_ploy, y)
end = time.perf_counter()
R2_score = r2_score(theta, x_ploy, y)
print(f'训练时间: {(end - start)*1000} ms')
print(f'theta:{theta}')
print(f'R2_score: {R2_score}')

y_hat_lst = []
for index in range(len(x)):
    y_hat = x_ploy[index][0] * theta[0] + x_ploy[index][1] * theta[1] + x_ploy[index][2] * theta[2]
    y_hat_lst.append(y_hat)

x2 = [i[1] for i in x_ploy]
plt.plot(x2, y_hat_lst, c='g', label='预测值')
plt.scatter(x2, y, c='r', label='真值')
plt.show()
