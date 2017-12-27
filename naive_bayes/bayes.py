import numpy as np
import random

# 文档集
def load_dataset():
    post_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['Mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']] # 发布的文档集
    label_list = [0, 1, 0, 1, 0, 1] # 1 represents insulting words, else 0 to be general
    return post_list, label_list

# 返回文档集对应词汇表
def create_vocabulary(data):
    vocabulary = set()
    for document in data:
        vocabulary = vocabulary.union(document)
    return list(vocabulary)

# 将指定文档转为向量(set-of-words model), 也可使用bag-of-words model
def to_vector(document, vocabulary):
    vector = [0]*len(vocabulary)
    for token in document:
        if token in vocabulary:
            vector[vocabulary.index(token)] = 1 # vector[vocabulary.index(token)] += 1
    return vector

def fit(train_matrix, categories):
    num_of_documents = train_matrix.shape[0]
    length_of_vocabulary = train_matrix.shape[1]
    probability_of_category1 = float(np.sum(categories)/num_of_documents) # 类别1的概率
    tokens_of_category0 = np.zeros(length_of_vocabulary) # 类别0中每一个词条(token)的个数
    tokens_of_category1 = np.zeros(length_of_vocabulary) # 类别1中每一个词条(token)的个数
    totals_of_category0 = 0. # 类别0中所有词条的数目
    totals_of_category1 = 0. # 类别1中所有词条的数目
    for index, document in enumerate(train_matrix):
        if categories[index] == 0:
            tokens_of_category0 += document
            totals_of_category0 += np.sum(document)
        else:
            tokens_of_category1 += document
            totals_of_category1 += np.sum(document)
    probabilities0 = tokens_of_category0/totals_of_category0
    probabilities1 = tokens_of_category1/totals_of_category1
    probabilities0[probabilities0==0] = 1./totals_of_category0/10. # 当前类别中某个词条不在字典中，则给其一个较小的概率值
    probabilities1[probabilities1==0] = 1./totals_of_category1/10.
    return probabilities0, probabilities1, probability_of_category1

def predict(docment_for_vector, probability0, probability1, probability1_of_category1):
    probability0 = np.log(probability0) # 避免下溢出
    probability1 = np.log(probability1)
    p0 = np.sum(docment_for_vector*probability0)+np.log(1.-probability1_of_category1)
    p1 = np.sum(docment_for_vector*probability1)+np.log(probability1_of_category1)
    if p0 > p1:
        return 0
    else:
        return 1

# def predict1(docment_for_vector, probability0, probability1, probability1_of_category1):
#     import functools as fun
#     p0 = fun.reduce(lambda x,y: x*y, docment_for_vector*probability0) * (1.-probability1_of_category1)
#     p1 = fun.reduce(lambda x,y: x*y, docment_for_vector*probability1) * probability1_of_category1
#     if p0 > p1:
#         return 0
#     else:
#         return 1

def test(document):
    post_list, label_list = load_dataset()
    voc = create_vocabulary(post_list)
    docment_for_vector = to_vector(document, voc)
    train_matrix = []
    for doc in post_list:
        train_matrix.append(to_vector(doc, voc))
    train_matrix = np.array(train_matrix)
    pred = predict(docment_for_vector, *fit(train_matrix, label_list))
    print('predict: %d' % pred)

# ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']-->0
# ['Mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him']-->0
# ['maybe', 'not', 'take', 'him', 'to', 'park', 'stupid']-->1
post_list, label_list = load_dataset()
doc_index = random.randint(0, len(post_list)-1)
target = label_list[doc_index]
# target = 0
print('target: %d' % target)
test_doc = post_list[doc_index]
# test_doc = ['dog', 'help', 'how', 'not', 'park', 'please', 'to', 'my', 'take']
test(test_doc)
