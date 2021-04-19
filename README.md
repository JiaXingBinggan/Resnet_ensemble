# Resnet_ensemble
ensemble learning for resnet

# Requirements

平台:Ubuntu 18.04.4
```
1.torch==1.4.0
2.torchvision==0.5.0
3.python==3.6.9
4.numpy==1.17.0
5.opencv-python==4.1.1.26
6.tqdm==4.46.0
7.thop==0.0.31
8.Cython==0.29.19
9.matplotlib==3.2.1
10.pycocotools==2.0.0
11.apex==0.1
12.DCNV2==0.1
```
**怎么装apex?**

按照如下顺序安装apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
如果上述步骤不行，可以试试下面的步骤
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

# 运行步骤
> 1. python3 train.py
> 2. python3 generate_preds.py
> 3. python3 vote.py
> 4. python3 ensemble.py

# 变种Bootstrap采样
> 自助法以自助采样法为基础，采用放回抽样的方法，从包含m个样本的数据集D中抽取m次，
组成训练集D'，然后数据集D中约有36.8%的样本未出现在D'中，于是自然而然可以用D\D'作为测试集
> 但是效果不佳
> 因此采取了从m个样本的数据集D中抽取2*m次，
组成训练集D''（训练集需去重），然后数据集D中约有13%-14%的样本未出现在D’中，再用D\D''作为测试集  
> **采样器为utils.py中BootStrap类**

# 集成策略
## 基本思路
> 1. 这30个基分类器里，先选效果最好的那个分类器a，再选出TOP N个效果好的学习器。
> 这N个分类器分别和a对比差异性，选择差异度最大的分类器b与a集成(投票)，得到新的分类器a'。
> 再在剩下的N-1个分类器里分别与a'计算差异度，取大的模c与a'集成，得到a'' ，...直到a''''的性能不再提升  
> 2. 该思路来源《面向点击率预测的集成学习关键算法研究》
## 改进思路
> 1. 因为这是多分类问题，因此每次直接选两个分类器来进行投票集成效果不佳。
> 采取的策略是：  
> 1）首先利用投票集成得到30个基分类器（N_i,i={1,2,...,30}）的投票集成预测结果，即为O，ACC值记为o。  
> 2）然后进入循环，并计算30个基分类器N_i与O的预测结果之间的皮尔逊相关系数P_i（或其他相似度度量）  
> 3）对P_i,i={1,2,...,N}进行排序，取前iter_ensemble_n个基分类器和O进行投票集成，并得到新的最优分类器O'，
> 如果ACC值o'>o，则继续迭代；反之则选择O为最终分类器。  
> 4）集成结束后，去掉P_i值最小的drop_ensemble_n个基分类器（这些基分类器不再参与集成），即P_i,i={1,2,...,N}，N=N-drop_ensemble_n
> 重复第2-3步骤。  
> 2. **代码位于ensemble.py第176至222行内**