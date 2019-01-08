### K近邻算法（knn）

#### 1、k近邻算法概述

kNN算法的模型就是整个训练数据集。当需要对一个未知数据实例进行预测时，kNN算法会在训练数据集中搜寻k个最相似实例。对k个最相似实例的属性进行归纳，将其作为对未知实例的预测。

- 优点：

  1、简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归；

  2、可用于数值型数据和离散型数据；

  3、对异常值不敏感 

- 缺点

  1、计算复杂性高；空间复杂性高；

  2、样本不平衡问题（即有些类别的样本数量很多，而其他样本数量很少）；

  3、无法给出数据的内在含义；

  4、高度数据相关，对极值敏感；

  5、维数灾难，随着维数的增加，看似相近的两个点之间的距离越来越大。

  

#### 2、k近邻的距离度量表示法

- 欧氏距离
- 曼哈顿距离
- 切比雪夫距离
- 闵可夫斯基距离
- 马氏距离
- 巴氏距离



#### 3、算法过程

算法过程：

（1）计算已知类别数据集中的点与当前点之间的距离；  

（2）按照距离递增次序排序；

（3）选取与当前点距离最小的K个点；  

（4）确定前K个点所在类别的出现频率；  

（5）返回前K个点出现频率最高的类别作为当前点的预测分类。



#### 4、scikit-learn中的k近邻算法

```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 加载手写字识别的数据集
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 将数据集拆分为训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# 调用分类器方法
knn_clf = KNeighborsClassifier()

# 训练模型
knn_clf.fit(X_train,y_train)

# ===================================
# 预测新的数据x
# 参数为矩阵，如果只有一个数据，需要转换
# X_predict = x.reshape(1,-1)
# 返回的是ndarray
# y_predict = knn_clf.predict(X_predict)
# ===================================

# 模型评价
knn_score = knn_clf.score(X_test,y_test)
print(knn_score)
```



####  5、超参数

- 最好的k

  ```python
  import numpy as np
  from sklearn import datasets
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import train_test_split
  
  # 加载手写字识别的数据集
  digits = datasets.load_digits()
  X = digits.data
  y = digits.target
  # 拆分数据集
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=666)
  
  best_method = ""
  best_score = 0.0
  best_k = -1
  for method in ['uniform','distance']:
      for k in range(1,11):
          knn_clf = KNeighborsClassifier(n_neighbors=k,weights=method)
          knn_clf.fit(X_train,y_train)
          score = knn_clf.score(X_test,y_test)
          if score > best_score:
              best_k = k
              best_score = score
              best_method = method
  
  print("best_method =",best_method)
  print("best_k =",best_k)
  print("best_score =",best_score)
  ```

- 搜索明可夫斯基距离相应的p

  ```python
  import numpy as np
  from sklearn import datasets
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import train_test_split
  
  # 加载手写字识别的数据集
  digits = datasets.load_digits()
  X = digits.data
  y = digits.target
  
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=666)
  
  best_p = -1
  best_score = 0.0
  best_k = -1
  for k in range(1,11):
      for p in range(1,6):
          knn_clf = KNeighborsClassifier(n_neighbors=k,weights='distance',p=p)
          knn_clf.fit(X_train,y_train)
          score = knn_clf.score(X_test,y_test)
          if score > best_score:
              best_k = k
              best_score = score
              best_p = p
  
  print("best_p =",p)
  print("best_k =",best_k)
  print("best_score =",best_score)
  # 也叫网格搜索 k*p
  ```

- 网格搜索

  ```python
  import numpy as np
  from sklearn import datasets
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import GridSearchCV
  
  # 加载手写字识别的数据集
  digits = datasets.load_digits()
  X = digits.data
  y = digits.target
  
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=666)
  
  param_grid =[
      {
          'weights':['uniform'],
          'n_neighbors':[i for i in range(1,11)]
      },
      {
          'weights':['distance'],
          'n_neighbors':[i for i in range(1,11)],
          'p':[i for i in range(1,6)]
      }
  ]
  knn_clf = KNeighborsClassifier()
  
  # n_job 可以利用多核来加速计算
  grid_search = GridSearchCV(knn_clf,param_grid)
  grid_search.fit(X_train,y_train)
  print(grid_search.best_estimator_)
  knn_clf = grid_search.best_estimator_
  knn_clf.predict(X_test)
  
  score = knn_clf.score(X_test,y_test)
  print(score)
  ```



#### 6、数字归一化

目的：将同所有数据映射到同一尺度内。

- 最值归一化 

  X = (x-x_min)/(x_max-x_min) 。把所有数据映射到0-1之间，这种方法适用于有明显边界的数据，比如学生的成绩在0-100之间，受outlier影响较大 ，比如收入问题，可能绝大部分样本在1W左右，对部分样本为100W的进行归一化的效果就不够明显。

  ```python
  import numpy as np
  from matplotlib import pyplot as plt
  x = np.random.randint(0,100,size=100)
  g = (x-np.min(x)) / (np.max(x) - np.min(x))
  print(g)
  
  X = np.random.randint(0,100,(50,2))
  X = np.array(X,dtype=float)
  X[:,0] = (X[:,0] - np.min(X[:,0])) /(np.max(X[:,0]) - np.min(X[:,0]))
  plt.scatter(X[:,0],X[:,1])
  plt.show()
  ```

  

- 均值方差归一化(Standardization)

  X = (X - X_mean) / S(方差)。把所有数据归一到均值为0方差为1的分布中。一般除非有特别明显的边界，我们都采用均值归一化。

  ```python
  import numpy as np
  from matplotlib import pyplot as plt
  
  X2 = np.random.randint(0,100,(50,2))
  X2 = np.array(X2,dtype=float)
  X2[:,0] = (X2[:,0] - np.mean(X2[:,0])) / np.std(X2[:,0])
  plt.scatter(X2[:,0],X2[:,1])
  plt.show()
  ```



#### 7、对测试数据集进行归一化处理

- 真实环境很有可能无法得到所有测试数据的均值和方差

- 对数据的归一化也是算法的一部分

  对于训练数据我们可以求出mean_train和std_train 对于测试数据 我们虽然也可以求得均值和方差，但一般也用（x_test-mean_train）/ std_train 后面进行预测时也使用训练数据集的均值和方差。

  ```python
  import numpy as np
  from sklearn import datasets
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.neighbors import KNeighborsClassifier
  
  iris = datasets.load_iris()
  X = iris.data
  y = iris.target
  
  X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=666)
  
  sd = StandardScaler()
  sd.fit(X_train)
  
  sd_mean = sd.mean_
  # 得到标准差
  sd_scale = sd.scale_
  
  X_train = sd.transform(X_train)
  X_test_std = sd.transform(X_test)
  
  knn_clf = KNeighborsClassifier(n_neighbors=3)
  knn_clf.fit(X_train,y_train)
  
  sd_score = knn_clf.score(X_test_std,y_test)
  base_score = knn_clf.score(X_test,y_test)
  
  print("测试数据集进行标准化处理之后的模型表现：%s" % sd_score)
  print("测试数据集未进行标准化处理的模型表现：%s" % base_score)
  ```




#### 5、思路扩展

- 回归问题：可以将本实现应用到一些回归问题（预测基于数值的属性）。对近邻实例的汇总可能涉及要预测属性的平均数或者中位数
- 归一化：当属性之间的度量单位不同时，很容易造成某些属性在距离度量层面成为主导因素。对于这类问题，你应该在相似性度量前将属性值都放缩到0-1范围内（称为归一化）。将模型升级以支持数据归一化。
- 多种距离度量：通常有许多距离度量方法可供选用，如果你愿意，甚至可以创造出针对特定领域的距离度量方法。实现替代的距离度量方法，例如曼哈顿距离(Manhattan distance)或向量点积(vector dot product)。

该算法还有很多扩展形式可以探索。这里给出两个扩展思路，包括基于距离权重的k-most相似性实例去预测以及更进一步的基于树形结构查找相似度去查找。

 