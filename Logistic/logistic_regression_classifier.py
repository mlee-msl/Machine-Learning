'''
Created on 2017年11月08日

@author: MLee
'''

import os
import numpy as np
import matplotlib.pyplot as plt

def create_dataset():
    nums = 300
    X0 = np.ones(nums)
    X1 = np.linspace(-8, 10, nums)
    X2 = 5./3.*X1 + 5.
    error = np.random.normal(1, 12, (nums,)) # 利用正态分布产生随机偏差
    _X2_ = X2 + error
    target = np.zeros(nums, dtype=np.int32)
    for i  in range(len(error)):
        if error[i] > 0: # positive instance
            target[i] = 1
    # print(target.dtype)
    data = np.column_stack((X0, X1, _X2_))
    if not os.path.exists('logistic_data.txt'):
        with open('logistic_data.txt', 'w') as fw:
            for i in range(len(data)):
                fw.write(','.join(map(str, data[i]))+','+str(target[i]))
                if i < len(data)-1:
                    fw.write('\n')

def load_dataset():
    data = [] # 实例特征值
    target = [] # 实例类别标签
    with open('logistic_data.txt', 'r') as fr:
        for row in fr.readlines():
            instance = row.strip().split(',')
            data.append(list(map(float, instance[:-1])))
            target.append(int(instance[-1]))
    return data, target

def load_dataset1():
    data = np.loadtxt('logistic_data.txt', dtype=np.float, delimiter=',', usecols=range(3))
    target = np.loadtxt('logistic_data.txt', dtype=np.int, delimiter=',', usecols=(-1,))
    return data, target

def sigmoid(Z):
    return 1.0/(1+np.exp(-Z))

# 梯度上升算法(每次使用整个数据集来更新weights)
def gradient_ascent(data, target):
    data = np.mat(data)
    target = np.mat(target).transpose()
    times = 126
    alpha = 0.001 # step length
    weights = np.ones((data.shape[1], 1)) # 初始化回归系数为1
    while times:
        predicts = sigmoid(data*weights)
        error = predicts - target # 样本偏差
        weights -= alpha*data.transpose()*error
        times -= 1
    return weights, stochastic_gradient_ascent(data, target), improved_stochastic_gradient_ascent(data, target)

# 随机梯度上升算法(每次使用一个实例来更新weights)
def stochastic_gradient_ascent(data, target):
    data = np.array(data)
    target = np.array(target)
    alpha = 0.01 # step length
    weights = np.ones((data.shape[1],))
    for index, instance in enumerate(data):
        predict = sigmoid(np.sum(instance*weights))
        error = target[index] - predict
        weights += alpha*instance*error
    return weights

# 增强随机梯度上升算法(每次使用随机实例以及变化学习率来更新weights)  
def improved_stochastic_gradient_ascent(data, target, num_of_iteration=126):
    data = np.array(data)
    target = np.array(target)
    weights = np.ones(data.shape[1])
    for i in range(num_of_iteration):
#         iteration_indices = random.sample(range(data.shape[0]), k=len(range(data.shape[0]))) # np.random.permutation(range(data.shape[0]))
        iteration_indices = np.random.permutation(range(data.shape[0]))
        for index in iteration_indices:
            alpha = 2./(1.+i+index)+0.05
            predict = sigmoid(np.sum(data[index]*weights))
            error = target[index] - predict
            weights += alpha*error*data[index]
    return weights


def fit(weight1, weight2, weight3):
    data, target = load_dataset()
    data = np.array(data)
    positive_xcoordinate = []
    positive_ycoordinate = []
    negative_xcoordinate = []
    negative_ycoordinate = []
    for index, instance in enumerate(data):
        if target[index] == 0: # negative
            negative_xcoordinate.append(instance[1])
            negative_ycoordinate.append(instance[2])
        else: # positive
            positive_xcoordinate.append(instance[1])
            positive_ycoordinate.append(instance[2])

    plt.figure(figsize=(9, 6))
    plt.suptitle('logistic_regression_classifier')
    plt.subplot(2, 2, 1)
    plt.title('target')
    plt.scatter(negative_xcoordinate, negative_ycoordinate, color='g', marker='*', label='negative')
    plt.scatter(positive_xcoordinate, positive_ycoordinate, color='b', marker='x', label='positive')
    plt.legend()
    X1 = np.linspace(-8, 10, 100)
    X2 = 5./3.*X1 + 5.0
    plt.plot(X1, X2, color='r')
    
    plt.subplot(222)
    plt.title('gradient_ascent')
    plt.scatter(negative_xcoordinate, negative_ycoordinate, color='g', marker='*', label='negative')
    plt.scatter(positive_xcoordinate, positive_ycoordinate, color='b', marker='x', label='positive')
    plt.legend()
    X1 = np.linspace(-8, 10, 100)
    X2 = (-weight1[0]-weight1[1]*X1)/weight1[2]
    plt.plot(X1, X2, color='r')

    plt.subplot(223)
    plt.title('stochastic_gradient_ascent')
    plt.scatter(negative_xcoordinate, negative_ycoordinate, color='g', marker='*', label='negative')
    plt.scatter(positive_xcoordinate, positive_ycoordinate, color='b', marker='x', label='positive')
    plt.legend()
    X1 = np.linspace(-8, 10, 100)
    X2 = (-weight2[0]-weight2[1]*X1)/weight2[2]
    plt.plot(X1, X2, color='r')
    
    plt.subplot(224)
    plt.title('improved_stochastic_gradient_ascent')
    plt.scatter(negative_xcoordinate, negative_ycoordinate, color='g', marker='*', label='negative')
    plt.scatter(positive_xcoordinate, positive_ycoordinate, color='b', marker='x', label='positive')
    plt.legend()
    X1 = np.linspace(-8, 10, 100)
    X2 = (-weight3[0]-weight3[1]*X1)/weight3[2]
    plt.plot(X1, X2, color='r')
    plt.show()

create_dataset()
fit(*gradient_ascent(*load_dataset()))
