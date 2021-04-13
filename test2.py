import numpy as np

'''
1. 自助法以自助采样法为基础，采用放回抽样的方法，从包含m个样本的数据集D中抽取m次，
组成训练集D'，然后数据集D中约有36.8%的样本未出现在D’中，于是我们用D\D‘作为测试集
2. 自助法在数据集较小，难以划分训练/测试集时很有用
3. 自助法能从初始数据中产生多个不同的训练集，这对集成学习等方法有很大的好处
4. 自助法产生的数据集改变了数据集的分布，会引入估计误差
'''

# 随机产生我们的数据集
x = np.random.randint(-10, 10, 10)  # 前两个参数表示范围，第三个参数表示个数
index = [i for i in range(len(x))]   # 数据集的下标
train_set = []  # 训练集
train_index = []  # 用于记录训练集各个元素的下标
test_set = []  # 测试集
test_index = []  # 用于记录测试集各个元素的下标

print(len(x))
# 进行m次放回抽样
for i in range(len(x)):
    print(np.floor(np.random.random()*len(x)))
    train_index.append(np.floor(np.random.random()*len(x)))
# 计算D\D'
test_index = list(set(index).difference(set(train_index)))

# 取数，产生训练/测试集
for i in range(len(train_index)):
    train_set.append(x[int(train_index[i])])  # 这里记得强制转换为int型，否则会报错
for i in range(len(test_index)):
    test_set.append(x[int(test_index[i])])

# 打印结果进行验证
print("data set: ", x)
print("train set: ", train_set)
print("test set: ", test_set)
