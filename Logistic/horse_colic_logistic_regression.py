from ccnu.ml.Horse_Colic import preprocessing as pre
import os
import numpy as np

def load_dataset(fname):
    if os.path.exists(fname):
        dataset = np.loadtxt(fname, dtype=np.float64, delimiter=' ')
        return dataset[:,:-1], dataset[:,-1].astype(np.int16)
    else:
        print('No such file or directory: \'%s\'' % fname)

# print(load_dataset('Horse_Colic_Training.txt'))

def sigmoid(z):
    return 1./(1.+np.exp(-np.longdouble(z)))

def stochastic_gradient_ascent(data, target, num_of_iterations=126):
    weights = np.ones(data.shape[1], dtype=np.float64)
    for i in range(num_of_iterations):
        iteration_indices = np.random.permutation(range(data.shape[0]))
        for j in iteration_indices:
#             alpha = 2./(2./(1.+j)+i)+0.01
            alpha = 0.001
            predict = sigmoid(np.sum(data[j]*weights))
            error = target[j] - predict
            weights += alpha * error * data[j]
    return weights


def predict(instance, weights):
    probability = sigmoid(np.sum(instance*weights))
    if probability > .5:
        return 1
    else:
        return 0
    
def get_accuracy(data, target, weights):
    accuracy_count = 0.
    for index, instance in enumerate(data):
        if predict(instance, weights) == target[index]:
            accuracy_count += 1
    return accuracy_count/data.shape[0]

def get_average_accuracy(data, target, times=10):
    data_training, target_training = load_dataset('Horse_Colic_Training.txt')
    nums = times
    sum_of_error = 0.
    while nums:
        nums -= 1
        weights = stochastic_gradient_ascent(data_training, target_training)
        sum_of_error += get_accuracy(data, target, weights)
    return '%.2f%%' % (sum_of_error/times*100.0)

pre.generate_dataset()
dataset_test = load_dataset('Horse_Colic_Test.txt')
if dataset_test is not None:
    acc = get_average_accuracy(*dataset_test, 6)
    print(acc)


