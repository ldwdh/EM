# EM
heros.csv中是王者荣耀中69名英雄的20个特征属性，EM.py中是具体的模型代码。该模型首先对数据进行了探索，采用热力图观察数据相关性，对数据进行降维，把原有的20个特征向量降为12个，然后再对数据进行规范化，最后使用GMM高斯混合模型进行聚类，并输出聚类结果。