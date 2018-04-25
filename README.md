# 深度学习

## 线性代数基础

### 视频课程

[[推荐]可视化理解：线性代数的本质 - 系列合集](https://www.bilibili.com/video/av6731067/)

### 课程笔记：

#### 向量

#### 基

#### 张成的空间

> 基张成的空间
>
> 列张成的空间（列空间）

#### 矩阵：线性变换

#### 线性变换：原点、平行、等距

#### 矩阵乘法：对空间进行线性变换

#### 线性变换复合：

#### 行列式：线性变换前后面积（体积）的缩放比例

> 行列式 > 0：翻转
>
> 行列式 = 零：矩阵的列必然线性相关（降维了）
>
> 行列式 < 负：未翻转

#### 逆矩阵：行列式不为0，则矩阵可逆（线性变换可逆）

> 为解**线性方程组**，而提出**逆矩阵**概念。
> $$
> a_{11}x _{1} + a_{12}x _{2} + \dots + a_{1n}x _{n} = k1\\
> a_{21}x _{1} + a_{22}x _{2} + \dots + a_{2n}x _{n} = k2 \\
> \dots \\
> a_{n1}x _{1} + a_{n2}x _{2} + \dots + a_{nn}x _{n} = k2
> $$
>
> $$
> \begin{bmatrix} 
> a_{11} & a_{12} & \dots & a_{1n} \\ 
> a_{21} & a_{22} & \dots & a_{2n} \\  
> \vdots & & & \vdots \\
> a_{n1} & a_{n2} & \dots & a_{nn} \\ 
> \end{bmatrix} 
> \begin{bmatrix} 
> x1 \\
> x2 \\
> \vdots \\
> xn
> \end{bmatrix} 
> = 
> \begin{bmatrix} 
> k1 \\
> k2 \\
> \vdots \\
> kn
> \end{bmatrix}
> $$
>
> $$
> A\vec x = \vec k
> $$
>
> 寻找向量$\vec x$，使得$\vec x$经过线性变换A之后，与向量$\vec k$重合。

#### 列空间：列张成的空间

#### 零空间

$$
\begin{bmatrix} 
a_{11} & a_{12} & \dots & a_{1n} \\ 
a_{21} & a_{22} & \dots & a_{2n} \\  
\vdots & & & \vdots \\
a_{n1} & a_{n2} & \dots & a_{nn} \\ 
\end{bmatrix} 
\begin{bmatrix} 
x1 \\
x2 \\
\vdots \\
xn
\end{bmatrix} 
= 
\begin{bmatrix} 
0 \\
0 \\
\vdots \\
0
\end{bmatrix}
$$

#### 秩

> 满秩
>
> 行满秩：矩阵的行向量线性无关
>
> 列满秩：矩阵的列向量线性无关
>
> 矩阵维度：矩阵的行数

#### 非方阵：行数 $\ne$ 列数

#### 点积/点乘：n纬 线性变换到 1维

> 点积 > 0：方向相同
>
> 点积 = 0：垂直
>
> 点积 < 0：方向相反

#### 对偶性

> 对偶向量：点乘时，等价的线性变换向量的转置

#### 叉积/叉乘

#### 基变换

> $\vec x' = A^{-1} M A  \vec x$
>
> 含义：将新空间的$\vec x$变换（$A$）到原空间，再在原空间执行线性变换M，再变换（$A^{-1}$）到新空间。
>
> $\vec x$：在新空间的向量
>
> A：新的基（新空间的基在原空间中的表示）
>
> M：线性变换
>
> $A^{-1}$：新的基的逆矩阵

#### 特征向量/特征值

> 特征向量：向量在线性变换后依然停留在该向量张成的空间中。
>
> 特征值：线性变换后特征向量的缩放值。
>
> 特征向量可能不存在。

>$A \vec v = \lambda \vec v$
>
>$\vec v$：A的特征向量
>
>$\lambda$：A的特征值

> 求$\lambda$，思路：
>
> $A \vec v = \lambda \vec v$
>
> $A \vec v = \lambda I \vec v$
>
> $A \vec v - \lambda I \vec v$  = $\vec 0$
>
> $(A - \lambda I) \vec v$  = $\vec 0$
>
> $det(A - \lambda I) = \vec 0$

>已知$\lambda$，求$\vec x$，思路：
>
>



### 参考

[行列式、向量、矩阵如何直观的理解？](https://www.zhihu.com/question/47467377/answer/106173563)



## 其他基础

### 高斯分布/正态分布

> 似然



## 斯坦福课程学习

[斯坦福大学公开课 ：机器学习课程](http://open.163.com/special/opencourse/machinelearning.html)

---

### 笔记：[第1课]机器学习的动机与应用

* Matlab / Octave：课程作业完成工具

  Matlab - 收费

  Octave - 免费



* 机器学习定义

   任务T
   评估P
   经验E

* 课程四大部份：

  1. 监督学习：有标准答案

     分类问题（分类算法）

  2.  学习理论：

  3. 无监督学习：无标准答案（从数据中挖掘有趣的结构）

  			聚类问题（聚类算法）
  			如“鸡尾酒会问题”-独立组件分析

  4. 强化学习：

			回报函数
	
支持向量机



* 回归问题
* 分类问题


[Octave_for_macOS下载 ](http://wiki.octave.org/Octave_for_macOS)

---

### 笔记：[第2课]监督学习应用.梯度下降


线性回归（一种参数学习算法）
	m：训练集大小
	x：输入
	y：输出
	(x, y)：一个样本

梯度下降算法
	局部最优问题
	批梯度下降算法
	随机梯度下降算法/增量梯度下降算法
	梯度上升算法
	
	矩阵的迹：n×n矩阵A的主对角线（从左上方至右下方的对角线）上各个元素的总和。
	矩阵转置：

正规方程组
	正规方程组：（使得最小二乘拟合问题可以不再用梯度下降算法进行迭代计算）
	最小二乘拟合问题：

---

### 笔记：[第3课]欠拟合与过拟合的概念


欠拟合
过拟合

防止欠拟合/过拟合，使用特征选择算法

参数学习算法：有固定数目的参数，以用来进行数据拟合的算法
非参数学习算法：参数数量会随着训练集级大小m线性增长的算法。（每次预测时，都需要重新拟合一次，计算量较大。使得此算法对于大型数据集更高效的方法是：KD tree by Andrew Moore）

局部加权回归（Loess）（一种非参数学习算法）

logistic回归
	logistic函数（sigmoid函数）

感知器算法

牛顿方法



参考：

[线性回归和逻辑回归](https://blog.csdn.net/u010692239/article/details/52345754)

---

### 笔记：[第4课]牛顿方法







# 概念

-   Graph：图，使用图来表示计算任务。
-   Session：上下文，图在此上下文中启动执行。
-   Tensor：张量，表示数据。
-   Variable：变量，维护状态。
-   Feed/Fetch：为任何操作（Operation，节点）写数据／读数据
-   Shape：


# CNN、RNN、DNN

感知机：



# 常用方法
* tf.Variable() : 变量
* tf.constant() : 常量
* tf.placeholder(） : 占位符
* tf.add() : 加
* tf.log() :
* tf.mul() :
* tf.matmul() : 矩阵乘法
* tf.reduce_sum() : 计算元素和
* tf.argmax() : 获取向量最大值的索引
* tf.cast() ： 映射到指定类型
* tf.equal() :
* tf.reduce_mean() : [张量不同数轴的平均值计算](https://www.cnblogs.com/yuzhuwei/p/6986171.html)
* tf.truncated_normal(shape, mean, stddev) ：产生满足正太分布的随机数（shape-张量维度，mean-均值，stddev-标准差），产生的随机数与均值的差距不会超过两倍的标准差。
* tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)：卷积函数 [TF-卷积函数 tf.nn.conv2d 介绍](https://www.cnblogs.com/qggg/p/6832342.html)
* tf.nn.softmax() : softmax模型
* sess = tf.Session() : 启动会话
* sess.run() : 执行图
* sess.close() : 关闭会话

# MNIST : Hellow World
* [MNIST机器学习入门](http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html)

* Softmax模型可以用来给不同对象分配概率
* Softmax回归

        图片 -> 标签
        
        每张图片：28*28=784像素 [1, 784]
        
        标签全集：数字0-9 [1, 10]
        
        训练数据集：60000张  [60000, 784] -> [60000, 10]
        
        测试数据集：10000张  [10000, 784]
    * 实现回归模型
    * 训练模型
        衡量模型好：
        衡量模型坏：成本coss/损失loss

        常见的coss函数：交叉熵（cross-entropy）
        交叉熵：用来衡量我们的预测用于描述真相的低效性。
        随机训练：使用一小部分随机数据进行训练。



---

参考资料：

[深层学习为何要“Deep”（上）](https://zhuanlan.zhihu.com/p/22888385)