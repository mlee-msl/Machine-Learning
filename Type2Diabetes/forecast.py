import random
import os
import sys
import xlrd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def excel_to_txt(file):
    table = xlrd.open_workbook(filename=file).sheet_by_name('Sheet1')
    fw = open('data.txt', 'w')
    for row in range(2, table.nrows-111):
        line_text = []
        for column in range(table.ncols):
            cell = str(table.cell_value(row, column)).strip()
            if len(cell) == 0:
                cell = '0'  # 空值补零
            line_text.append(cell)
        fw.write(','.join(line_text))
        fw.write('\n')
    fw.close()


def converter1(chromosome):
    if chromosome == b'X':
        return '23' # sex chromosome
    return chromosome # autosome


def converter2(outcome):
    if outcome == 'T2D'.encode(encoding='utf-8'):  # T2D
        return 1
    else:
        return 0  # Others


def normalization(data):
    min_features = np.min(data, axis=0)
    max_features = np.max(data, axis=0)
    normalized_data = (data - min_features) / (max_features - min_features)
    return normalized_data


def load_original_dataset(file):
    names = ','.join(map(str, range(21)))
    usecols = tuple(filter(lambda name: name != '0', names.split(',')))
    tmp = np.genfromtxt(fname=file, dtype=np.float, delimiter=',', converters={'1': converter1, '20': converter2}, usecols=usecols, names=names)
    dataset = np.ones((tmp.shape[0], 20), dtype=np.float)
    for i in range(tmp.shape[0]):
        dataset[i, :] = list(tmp[i])
    data, target = dataset[:, :-1].astype(dtype=np.float), dataset[:, -1].astype(dtype=np.int)
    data[:, 1] = normalization(data[:, 1])
    return data, target


def save_train_test_split(data, target, test_size=0.3):
    # normalization
    data[:, 1] = normalization(data[:, 1])
    indices = set(range(data.shape[0]))
    test_indices = random.sample(indices, int(data.shape[0] * test_size))
    training_indices = random.sample(indices.difference(test_indices), data.shape[0] - len(test_indices))
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
    # save all data(training & test)
    np.savetxt('training.txt', np.concatenate((training_data, training_target.reshape(training_data.shape[0], 1)), axis=1), fmt='%s', delimiter=',')
    np.savetxt('test.txt', np.concatenate((test_data, test_target.reshape(test_data.shape[0], 1)), axis=1), fmt='%s', delimiter=',')


def load_dataset(fname):
    if os.path.exists(fname):
        dataset = np.loadtxt(fname, dtype=np.float, delimiter=',')
        return dataset[:, :-1], dataset[:, -1].astype(np.int)
    else:
        print('No such file or directory: \'%s\'' % fname)


def main():
    excel_to_txt('data.xlsx')
    save_train_test_split(*load_original_dataset('data.txt'))
    data_training, target_training = load_dataset('training.txt')
    data_test, target_test = load_dataset('test.txt')
    data = StandardScaler()
    data_training = data.fit_transform(data_training)
    data_test = data.transform(data_test)
    lr = LogisticRegression()
    lr.fit(data_training, target_training)
    print('*'*10, 'LogisticRegression', '*'*10)
    print('accuracy: ', '%.2f%%' % (lr.score(data_test, target_test)*100.0))
    lr_predict = lr.predict(data_test)
    print(classification_report(target_test, lr_predict, target_names=['Others', 'T2D']))
    svc = SVC()
    svc.fit(data_training, target_training)
    print('*' * 10, 'SVM', '*' * 10)
    print('accuracy: ', '%.2f%%' % (svc.score(data_test, target_test)*100.0))
    svc_predict = svc.predict(data_test)
    print(classification_report(target_test, svc_predict, target_names=['Others', 'T2D']))
    rfc = RandomForestClassifier()
    rfc.fit(data_training, target_training)
    print('*' * 10, 'RandomForest', '*' * 10)
    print('accuracy: ', '%.2f%%' % (rfc.score(data_test, target_test)*100.0))
    rfc_predict = rfc.predict(data_test)
    print(classification_report(target_test, rfc_predict, target_names=['Others', 'T2D']))
    dtc = DecisionTreeClassifier()
    dtc.fit(data_training, target_training)
    print('*' * 10, 'DecisionTree', '*' * 10)
    print('accuracy: ', '%.2f%%' % (dtc.score(data_test, target_test)*100.0))
    dtc_predict = dtc.predict(data_test)
    print(classification_report(target_test, dtc_predict, target_names=['Others', 'T2D']))


if __name__ == '__main__':
    with open('report.txt', 'w') as fw:
        sys.stdout = fw
        main()
