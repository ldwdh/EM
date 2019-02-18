import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
#读取数据
data_ori=pd.read_csv('heros.csv',encoding='gb18030')
#可视化分析
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
corr=data_ori[list(data_ori)].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True)
plt.show()
#观察热力图后对属性进行降维
features_remain = ['最大生命','初始生命','最大法力','最高物攻','初始物攻','最大物防','初始物防','最大每5秒回血','最大每5秒回蓝','初始每5秒回蓝','最大攻速','攻击范围']
data=data_ori[features_remain]
#数据变换
data['最大攻速']=data['最大攻速'].apply(lambda x:float(x.strip('%'))/100)
data['攻击范围']=data['攻击范围'].map({'远程':1,'近战':0})
#数据规范化
ss=StandardScaler()
data=ss.fit_transform(data)
#构造GMM聚类
gmm=GaussianMixture(n_components=30,covariance_type='full')
gmm.fit(data)
#训练数据
prediction=gmm.predict(data)
data_ori.insert(0,'分组',prediction)
data_ori.to_csv('hero_out.csv',index=False,sep=',')
