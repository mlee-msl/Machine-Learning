from urllib import request
import random
import numpy as np


def converter(label):
    if label == b'Iris-setosa':
        return 0
    elif label == b'Iris-versicolor':
        return 1
    else: # Iris-virginica
        return 2

def load_dataset(url):
    response = request.urlopen(url)
    dataset = np.loadtxt(fname=response, dtype=np.float, delimiter=',', converters={4: converter})
    data, target = dataset[:,:-1].astype(dtype=np.float), dataset[:,-1]
    return data, target

def normalization(data):
    min_features = np.min(data, axis=0)
    max_features = np.max(data, axis=0)
    normalized_data = (data - np.tile(min_features, (data.shape[0], 1)))/(max_features-min_features)
    return normalized_data
    
def train_test_split(data, target, test_size=0.2):
    indices = set(range(data.shape[0]))
    test_indices = random.sample(indices, int(data.shape[0]*test_size))
    training_indices = random.sample(indices.difference(test_indices), data.shape[0]-len(test_indices))
    training_data = np.zeros((len(training_indices), data.shape[1]))
    training_target = np.zeros((len(training_indices),))
    test_data = np.zeros((len(test_indices), data.shape[1]))
    test_target = np.zeros((len(test_indices),))
    for i, index in enumerate(training_indices):
        training_data[i] = data[index]
        training_target[i] = target[index]
    for i, index in enumerate(test_indices):
        test_data[i] = data[index]
        test_target[i] = target[index]
    return training_data, training_target, test_data, test_target    
    
def fit(data, target, k=4):
    irisSetosa_data = []
    irisVersicolor_data = []
    irisVirginica_data = []
    for index, label in enumerate(target):
        if label == 0:
            irisSetosa_data.append(data[index])
        elif label == 1:
            irisVersicolor_data.append(data[index])
        else:
            irisVirginica_data.append(data[index])
    irisSetosa_center = np.mean(np.array(irisSetosa_data), axis=0)
    irisVersicolor_center = np.mean(np.array(irisVersicolor_data), axis=0)
    irisVirginica_center = np.mean(np.array(irisVirginica_data), axis=0)
    dis0 = np.sqrt(np.sum((np.array(irisSetosa_data)-irisSetosa_center)**2, axis=1))
    distance_indices0 = np.argsort(dis0)
    s0 = 0.
    for i in [-1*i for i in range(1, k+1)]:
        s0 += dis0[distance_indices0[i]]
    irisSetosa_bias = s0/k
    dis1 = np.sqrt(np.sum((np.array(irisVersicolor_data)-irisVersicolor_center)**2, axis=1))
    distance_indices1 = np.argsort(dis1)
    s1 = 0.
    for i in [-1*i for i in range(1, k+1)]:
        s1 += dis1[distance_indices1[i]]
    irisVersicolor_bias = s1/k
    dis2 = np.sqrt(np.sum((np.array(irisVirginica_data)-irisVirginica_center)**2, axis=1))
    distance_indices2 = np.argsort(dis2)
    s2 = 0.
    for i in [-1*i for i in range(1, k+1)]:
        s2 += dis2[distance_indices2[i]]
    irisVirginica_bias = s2/k
    return (irisSetosa_center, irisSetosa_bias), (irisVersicolor_center, irisVersicolor_bias), (irisVirginica_center, irisVirginica_bias)

import operator
def knn(data, target, test_data, k=7):
    predicts = []
    target = target.astype(int)
    for ith_test_data in test_data:
        distances = np.sum((ith_test_data-data)**2, axis=1)**.5
        distances_indices = np.argsort(distances)
        class_count = {}
        for i in range(k):
            class_count[target[distances_indices[i]]] = class_count.get(target[distances_indices[i]], 0) + 1
#         predicts.append(sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)[0][0])
        predicts.append([class_count.get(0, 0), class_count.get(1, 0), class_count.get(2, 0)])
    return np.array(predicts)

def predict(classes, test_data):
    distances = np.zeros((test_data.shape[0], len(classes)))
    for i, clazz in enumerate(np.array(classes)[:,0]):
        distances[:,i] = np.sum((np.tile(clazz, (test_data.shape[0], 1)) - test_data)**2, axis=1)**.5
    return distances.argmin(axis=1)

def predict1(classes, test_data):
    distances = np.zeros((test_data.shape[0], len(classes)))
    classes = np.array(classes)
    for i, clazz in enumerate(classes):
        tmp = np.sum((clazz[0] - test_data)**2, axis=1)**.5
        distances[:,i] = np.array(list(map(lambda x: max(0, x), tmp-clazz[1])))
    return distances.argmin(axis=1)

def predict2(classes, test_data, knn, theta=(.1,.2)):
    distances = np.zeros((test_data.shape[0], len(classes)))
    classes = np.array(classes)
    for i, clazz in enumerate(classes):
        h = np.sum((clazz[0] - test_data)**2, axis=1)**.5
        l = np.array(list(map(lambda x: max(0, x), h-clazz[1])))
        l = min(h)+(l-min(l))/(max(l)-min(l))*(max(h)-min(h))
        knn[:,i] = min(h)+np.abs((knn[:,i]-max(knn[:,i])))/(max(knn[:,i])-min(knn[:,i]))*(max(h)-min(h))
        knn_indices = np.argsort(knn)
        for ii in range(int(np.ceil(len(knn)/2))):
            knn[knn_indices[ii],i], knn[knn_indices[(-1)*(ii+1)],i] = knn[knn_indices[(-1)*(ii+1)],i], knn[knn_indices[ii],i]
        distances[:,i] = theta[0]*l+theta[1]*h+(1-sum(theta))*knn[:,i]
    return distances.argmin(axis=1)

def get_accuracy(predicts, test_target):
    count_errors = sum([predict != target for predict, target in zip(predicts, test_target)])
    return '%.2f%%' % ((1-count_errors/len(test_target))*100.0)
    
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
training_data, training_target, test_data, test_target = train_test_split(*load_dataset(url))
print(test_target.astype(np.int))
print()
classes = fit(training_data, training_target)
predicts_knn = knn(training_data, training_target, test_data).argmax(axis=1)
print(np.array(predicts_knn).astype(int))
print(get_accuracy(predicts_knn, test_target))

predicts = predict(classes, test_data)
print(predicts)
print(get_accuracy(predicts, test_target))
predicts1 = predict1(classes, test_data)
print(predicts1)
print(get_accuracy(predicts1, test_target))
predicts2 = predict2(classes, test_data, knn(training_data, training_target, test_data))
print(predicts2)
print(get_accuracy(predicts2, test_target))
