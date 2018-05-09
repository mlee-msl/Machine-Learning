import numpy as np


def personal_rank(bipartite_graph, alpha, user, epochs):
    rank = dict()
    rank = {x: 0 for x in bipartite_graph.keys()}  # 所有节点的初始化访问概率为0
    rank[user] = 1  # 但我们打算将要推荐的用户的初始化的访问概率设为1
    # 开始迭代
    for k in range(epochs):
        tmp = {x: 0 for x in bipartite_graph.keys()}
        # 取二部图中所有的节点i, 以及所有的以节点i为起点的（终点j，权重wij）集合ri, 注，此处所有权重均为1，只用来表示用户与商品是否有行为关系
        for i, ri in bipartite_graph.items():
            for j, wij in ri.items():
                # len(ri)表示以节点i为起点的边的条数，rank[i]表示节点i的PR值，并以alpha的概率从节点j继续游走
                tmp[j] += alpha * (rank[i] / (1.0 * len(ri)))
                # 若当前游走的节点为user(指定推荐者)，那么我们以（1-alpha）的概率选择重新以节点j重新开始游走，此处j为user时，概率为1，其他为0，注，PageRank算法每一个节点游走的概率一样且为1.0/（1.0*|V|）,|V|为节点总数
                tmp[j] += (1 - alpha) * (1 if j == user else 0)
        # 计算前后两次所有节点的PR值偏差和的差异大小
        diff = [abs(tmp.get(key)-rank.get(key)) for key in rank.keys()]
        if sum(diff) <= 0.0001:
            break
        rank = tmp
        # 输出每次迭代后各个节点的PR值
        print('iter: ' + str(k) + "\t", end='')
        for key, value in rank.items():
            print("%s:%.3f  \t" % (key, value), end='')
        print()
    return rank


def mat_to_bipartite_graph(node, matrix):
    bipartite_graph = {}
    matrix = np.array(matrix)
    M, N  = matrix.shape
    for m in range(M):
        tmp = {}
        for n in range(N):
            if matrix[m, n] != 0.0:
                tmp[node[n]] = matrix[m, n]
        bipartite_graph[node[m]] = tmp
    return bipartite_graph


if __name__ == '__main__':
    node = ['A', 'B', 'C', 'a', 'b', 'c', 'd']
    matrix = [[0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 1, 1],
              [1, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0, 0]]
    bipartite_graph = mat_to_bipartite_graph(node, matrix)
    # print(bipartite_graph)
    # bipartite_graph = {'A': {'a': 1, 'c': 1}, 'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
    #                    'C': {'c': 1, 'd': 1}, 'a': {'A': 1, 'B': 1},
    #                    'b': {'B': 1}, 'c': {'A': 1, 'B': 1, 'C': 1},
    #                    'd': {'B': 1, 'C': 1}}
    PR = personal_rank(bipartite_graph, 0.85, 'A', 126)  # 以alpha的概率从当前节点继续游走，并以`A`作为将要推荐用户，迭代126次
    print(PR)
