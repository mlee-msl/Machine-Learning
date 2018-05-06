import numpy as np

#  original_matrix --> P*Q=R
def matrix_factorization(original_matrix, K, alpha, beta, epochs):
    '''
    :param original_matrix(mat): 原始矩阵
    :param K(int): 分解矩阵中间维度
    :param alpha(float): 学习率
    :param beta(float): 惩罚性系数
    :param epochs(int): 最大迭代次数
    :return: 分解后的两个矩阵P,Q
    '''
    original_matrix = np.mat(original_matrix)
    M, N = original_matrix.shape
    P = np.mat(np.random.random((M, K)))
    Q = np.mat(np.random.random((K, N)))
    loss = 1.0
    epoch = 0
    while loss >= 0.001 and epoch <= epochs:
        for m in range(M):
            for n in range(N):
                if original_matrix[m, n] > 0:  # 非缺失值
                    r = original_matrix[m, n]
                    r_ = 0  # R[m, n]
                    for k in range(K):  # 计算[m, n]位置的误差
                        r_ += P[m, k]*Q[k, n]
                    e = r - r_
                    for k in range(K):  # 更新P[m, :]与Q[:, n]的值
                        P[m, k] += alpha*(2*e*Q[k, n]-beta*P[m, k])
                        Q[k, n] += alpha*(2*e*P[m, k]-beta*Q[k, n])
        loss = 0.0
        for m in range(M):
            for n in range(N):
                if original_matrix[m, n] > 0:
                    r = original_matrix[m, n]
                    r_ = 0.0
                    regularization = 0.0
                    for k in range(K):
                        r_ += P[m, k]*Q[k, n]  # 偏差
                        regularization += P[m, k]**2+Q[k, n]**2  # L2正则化
                    e = r - r_
                    loss += e**2 + (beta/2)*regularization  # 总损失
        # if epoch % 200 == 0:
        #     print('epoch:{}, loss: {}'.format(epoch, loss))
        epoch += 1
    return P, Q


if __name__ == '__main__':
    low, high = 1, 10
    size = 10, 8
    original_matrix = np.random.randint(low=low, high=high, size=size)
    missing_rate = 0.3
    n_counts = int(size[0]*size[1]*missing_rate)
    while n_counts:
        pos = np.random.randint(size[0]*size[1])
        row, col = pos//size[1], pos%size[1]
        if original_matrix[row, col] != 0:
            original_matrix[row, col] = 0
            n_counts -= 1
    # print(original_matrix)
    P, Q = matrix_factorization(original_matrix, 6, 0.004, 0.02, 12000)
    print(np.dot(P, Q))
