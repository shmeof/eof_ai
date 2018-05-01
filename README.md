# 深度学习

## 线性代数基础

### 视频课程

[[推荐]可视化理解：线性代数的本质 - 系列合集](https://www.bilibili.com/video/av6731067/)

### 课程笔记：

#### 向量

#### 基

基：一般看矩阵的列。

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

> 行列式与所选坐标系无关。	

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

>关注概念：特征向量/特征值
>$$
>\begin{bmatrix} 
>a_{11} & a_{12} & \dots & a_{1n} \\ 
>a_{21} & a_{22} & \dots & a_{2n} \\  
>\vdots & & & \vdots \\
>a_{n1} & a_{n2} & \dots & a_{nn} \\ 
>\end{bmatrix} 
>\begin{bmatrix} 
>x1 \\
>x2 \\
>\vdots \\
>xn
>\end{bmatrix} 
>= 
>\begin{bmatrix} 
>0 \\
>0 \\
>\vdots \\
>0
>\end{bmatrix}
>$$
>

#### 秩

> 满秩：
>
> 行满秩：矩阵的行向量线性无关
>
> 列满秩：矩阵的列向量线性无关
>
> 矩阵维度：矩阵的行数
>
> 奇异矩阵：矩阵不满秩
>
> 非奇异矩阵：矩阵满秩

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
>
> 特征向量与所选坐标系无关。

>$A \vec v = \lambda \vec v$
>
>$\vec v$：A的特征向量
>
>$\lambda$：A的特征值

> 求$\lambda$，思路：
>
> $A \vec v = \lambda \vec v$（$\vec v \ne \vec 0$ ）
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
>解线性方程组（关注概念：零空间）
>$$
>\begin{bmatrix} 
>a_{11}-\lambda & a_{12} & \dots & a_{1n} \\ 
>a_{21} & a_{22}-\lambda & \dots & a_{2n} \\  
>\vdots & & & \vdots \\
>a_{n1} & a_{n2} & \dots & a_{nn}-\lambda \\ 
>\end{bmatrix} 
>\begin{bmatrix} 
>x1 \\
>x2 \\
>\vdots \\
>xn
>\end{bmatrix} 
>= 
>\begin{bmatrix} 
>0 \\
>0 \\
>\vdots \\
>0
>\end{bmatrix}
>$$
>

#### 对角矩阵

> 对角矩阵：所有基向量都是特征向量，对角元是它们对应的特征值。
>
> 对角矩阵在很多计算方面都更容易处理。

#### 特征基

> 特征基：若矩阵A有多个特征向量，且其中存在可以张成全空间的一组向量集合，则可选取这样的集合作为空间的基。（这组特征基会使得矩阵运算变简单）
>
> 诸如计算$A^{n}$等的复杂运算，可先变换到特征基，在那个空间计算后，再转换回标准坐标系。

#### 抽象向量空间

> 函数

> 线性的严格定义：可加性、成比例（一阶齐次）

#### 基函数

> 基是一个函数的情况。

> 函数求导 与 矩阵乘法 的关系（举例：多项式）

> 线性变换 - 线性算子
>
> 点积 - 内积
>
> 特征向量 - 特征函数

> 向量加法和数乘的8条规则（公理）



### 参考

[行列式、向量、矩阵如何直观的理解？](https://www.zhihu.com/question/47467377/answer/106173563)



## 概率论基础

### 课程

掌握机器学习数学基础之概率统计（重点知识）](https://zhuanlan.zhihu.com/p/30314229?utm_medium=social&utm_source=wechat_session)

### 笔记

#### 频率学派／贝叶斯学派

> 频率学派：事件数据
>
> 贝叶斯学派：先验知识 + 事件数据

#### 随机变量（离散／连续）

> X：随机变量
>
> x：随机变量X的取值

#### 概率分布

#### 条件概率

> P(A|B)：给定条件B，事件A发生的概率。
>
> A与B是独立事件：P(AB) = P(A)P(B) 
>
> A与B非独立事件：P(AB) = P(A)P(B|A)

#### 链式法则 ／ 乘法法则

>  p(a,b,c) = p(a|b, c) p(b|c) p(c)

#### 联合概率

> P(AB)：事件A和事件B同时发生的概率。

#### 全概率公式

#### 边缘概率

？

#### 离散型随机变量

#### 连续型随机变量

#### 独立性



#### 条件独立性



#### 期望

#### 方差

#### 协方差

#### 相关系数



#### 概率分布-离散（0-1分布）

> 

#### 概率分布-离散（几何分布）

> 

#### 概率分布-离散（二项分布）

> 

#### 概率分布-离散（泊松分布）

> 

#### 概率分布-连续（均匀分布）

> 

#### 概率分布-连续（高斯分布／正态分布）

> 

#### 概率分布-连续（指数分布）

> 指数分布：事件的是时间间隔概率。
>
> 重要特征：无记忆性。
>
> 泊松分布 => 指数分布

#### 概率分布-连续（拉普拉斯分布）

>

#### 贝叶斯定理

> 贝叶斯定理：一种“根据数据集内容的变化而更新假设概率”的方法。
>
> $P(B|A)= \frac {P(A|B)P(B)}{P(A)}$
>
> P(B|A)：后验概率。
>
> P(B)：先验概率。
>
> P(A|B)：似然度。
>
> P(A)：标准化常量。

#### 中心极限定理

> 中心极限定理：大量相互独立的随机变量，其均值的分布以正态分布为极限。

#### 极大似然估计

> 极大似然估计：利用已知样本结果，反推最有可能（最大概率）导致这样结果的参数值。
>
> 解释：已知样本既然已经发生，则此组样本的概率乘积应该是最大的。
>
> 例子：
>
> > 一个袋子里放有：黑球，白球
> >
> > 每次取出一个球，得知黑白后放回袋子
> >
> > 取10次，结果：黑球8次，白球2次
> >
> > 问：袋子里的黑球白球比例最有可能是多少？
> >
> > > 答：设P(球=黑)=p，则对应该结果的概率为$P(黑=8) =p^8(1-p)^2$
> > >
> > > 使用极大似然估计思考：8黑2白的结果既然已经发生，说明这个结果是所有可能结果（包括已发生和未发生的结果）中概率最大的，所以使得$P(黑=8)$取到最大值的p即为所求。
>
> 公式：
>
> > 

#### 最大后验估计

> 

#### 独立同分布

> 独立
>
> 同分布
>
> 独立同分布



## 机器学习基础

[斯坦福大学公开课 ：机器学习课程](http://open.163.com/special/opencourse/machinelearning.html)

[Kivy-CN/Stanford-CS-229-CN: A Chinese Translation of Stanford CS229 notes 斯坦福机器学习CS229课程讲义的中文翻译](https://github.com/Kivy-CN/Stanford-CS-229-CN)

### 机器学习三要素

模型、策略、算法

### 线性回归

#### 求$\vec \theta$，迭代方法（慢）-最小均方算法（LMS）

$\begin{cases}线性函数：h(x)=\vec \theta^T\vec x\\成本函数：J(\theta)= \frac 12 \sum^m_{i=1}(H_\theta(x^{(i)})-y^{(i)})^2\\梯度下降：\theta_j := \theta_j - \alpha \frac \partial {\partial\theta_J}J(\theta)\end{cases}$ $\Rightarrow$ 最小均方算法（LMS）：$\theta_j := \theta_j + \alpha (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}$

$梯度下降\begin{cases}批量梯度下降\\增量梯度下降\end{cases}$

#### 求$\vec \theta$，非迭代方法（快）-法线方程

法线方程：$X^TX\theta=X^T \vec y$

使得$J(\vec \theta)$最小的$\theta=(X^TX)^-1X^T\vec y$

#### 最小二乘法得到的$\theta$和最大似然法得到的$\theta$是一致的

### 逻辑回归

#### 逻辑函数／双弯曲S型函数（sigmoid）

 $h_\theta(x) = g(\theta^T x) = \frac  1{1+e^{-\theta^Tx}}$

sigmoid：$g(z)= \frac 1 {1+e^{-z}}$

$\begin{aligned}\dot{g}(z) = g(z)(1-g(z))\\\end{aligned}$

#### 求$\vec \theta$，慢，梯度上升

$\begin{cases}逻辑函数：g(z)= \frac 1 {1+e^{-z}} \\ 似然函数：\begin{aligned}L(\theta) &= p(\vec{y}| X; \theta)\\&= \prod^m_{i=1}  p(y^{(i)}| x^{(i)}; \theta)\\&= \prod^m_{i=1} (h_\theta (x^{(i)}))^{y^{(i)}}(1-h_\theta (x^{(i)}))^{1-y^{(i)}} \\\end{aligned} \\ 梯度上升：\begin{aligned}\frac  {\partial}{\partial \theta_j} l(\theta) &=(y\times \frac  1 {g(\theta ^T x)}  - (1y)\times\frac  1 {1- g(\theta ^T x)}   )\frac  {\partial}{\partial \theta_j}g(\theta ^Tx) \\&= (y\times \frac  1 {g(\theta ^T x)}  - (1y)\times \frac  1 {1- g(\theta ^T x)}   )  g(\theta^Tx)(1-g(\theta^Tx)) \frac  {\partial}{\partial \theta_j}\theta ^Tx \\&= (y(1-g(\theta^Tx) ) -(1-y) g(\theta^Tx)) x_j\\&= (y-h_\theta(x))x_j\end{aligned}\end{cases}$$\Rightarrow$ $\theta_j := \theta_j + \alpha (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}$

#### 求$\vec \theta$，快，牛顿法（一维）/牛顿-拉普森法（多维）

$\begin{cases}逻辑函数：g(z)= \frac 1 {1+e^{-z}} \\ 似然函数：\begin{aligned}L(\theta) &= p(\vec{y}| X; \theta)\\&= \prod^m_{i=1}  p(y^{(i)}| x^{(i)}; \theta)\\&= \prod^m_{i=1} (h_\theta (x^{(i)}))^{y^{(i)}}(1-h_\theta (x^{(i)}))^{1-y^{(i)}} \\\end{aligned} \\ 牛顿-拉普森法：\theta := \theta - H^{-1}\nabla_\theta l(\theta)，H_{ij}= \frac {\partial^2 l(\theta)}{\partial \theta_i \partial \theta_j}\end{cases}$$\Rightarrow$求得最大$l(\theta)$及其对应的$\theta$(Fisher评分）

### 广义线性模型

#### 指数族

指数族：可用以下描述的分布。

$ p(y;\eta) =b(y)exp(\eta^TT(y)-a(\eta))$

#### 广义线性模型特例：普通最小二乘法



#### 广义线性模型特例：逻辑回归



#### 广义线性模型特例：Softmax回归



#### 广义线性模型特例：多项式分布



#### 广义线性模型特例：泊松分布



#### 广义线性模型特例：$\beta$和狄利克雷分布



#### 广义线性模型特例：$\gamma$和指数分布



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



## Tensorflow

### 概念

-   Graph：图，使用图来表示计算任务。
-   Session：上下文，图在此上下文中启动执行。
-   Tensor：张量，表示数据。
-   Variable：变量，维护状态。
-   Feed/Fetch：为任何操作（Operation，节点）写数据／读数据
-   Shape：


### CNN、RNN、DNN

感知机：



### 常用方法

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

### MNIST : Hellow World

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