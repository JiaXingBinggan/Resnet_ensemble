import random

'''
1. 自助法以自助采样法为基础，采用放回抽样的方法，从包含m个样本的数据集D中抽取m次，
组成训练集D'，然后数据集D中约有36.8%的样本未出现在D’中，于是我们用D\D‘作为测试集
2. 自助法在数据集较小，难以划分训练/测试集时很有用
3. 自助法能从初始数据中产生多个不同的训练集，这对集成学习等方法有很大的好处
4. 自助法产生的数据集改变了数据集的分布，会引入估计误差
'''

class BootStrap(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sampling(self):
        _slice = []
        while len(_slice) < self.n_samples:
            p = random.randrange(0, self.n_samples)
            _slice.append(p)
        return _slice


if __name__ == '__main__':
    random.seed(2021)
    boostrap = BootStrap(50000)
    for i in range(30):
        train_slice = boostrap.sampling()
        # print(len(train_slice))
        test_slice = list(set(list(range(50000))).difference(set(train_slice)))
        print(test_slice)