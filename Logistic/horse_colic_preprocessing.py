import os
from urllib import request
import numpy as np

def get_raw_dataset(fname, filename):
    if not os.path.exists(filename):
        response = request.urlopen(fname)
        raw_dataset = np.genfromtxt(response, dtype=np.str, delimiter=' ',)
        np.savetxt(filename, raw_dataset, fmt='%s')
    
def converter(outcome):
    if outcome == '1'.encode(encoding='utf_8'): # lived
        return 1
    else:
        return 0 # died or was euthanized
    
def get_processed_dataset(fname, filename):
    names = ','.join(map(str, range(28))) # the names of columns
    usecols = list(filter(lambda name: name != '2' and int(name) < 23, names.split(','))) # extract the specified columns
    if os.path.exists(fname):
        processed_dataset = np.genfromtxt(fname, dtype=None, delimiter=' ', converters={'22': converter}, missing_values='?', filling_values=0, usecols=usecols, names=names)
        np.savetxt(filename, processed_dataset, fmt='%s')

def generate_dataset():
    # 得到训练集的原始数据集
    get_raw_dataset(fname='http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data', filename='Raw_Horse_Colic_Training.txt')
    # 得到测试集的原始数据集
    get_raw_dataset(fname='http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.test', filename='Raw_Horse_Colic_Test.txt')
    # 由原始训练数据集得到'干净'的训练数据集
    get_processed_dataset('Raw_Horse_Colic_Training.txt', 'Horse_Colic_Training.txt')
    # 由原始测试数据集得到'干净'的测试数据集
    get_processed_dataset('Raw_Horse_Colic_Test.txt', 'Horse_Colic_Test.txt')

