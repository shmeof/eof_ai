# 人工智能

## 线性代数基础

### 课程

[[推荐]可视化理解：线性代数的本质 - 系列合集](https://www.bilibili.com/video/av6731067/)

[行列式、向量、矩阵如何直观的理解？](https://www.zhihu.com/question/47467377/answer/106173563)

### 笔记

#### 向量：

#### 基：矩阵的列

> 基：矩阵的列。

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

#### 点乘／点积／内积：n纬 线性变换到 1维

> 点积 > 0：方向相同
>
> 点积 = 0：垂直
>
> 点积 < 0：方向相反

#### 对偶性

> 对偶向量：点乘时，等价的线性变换向量的转置

#### 叉积／叉乘

> 

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

#### 特征向量／特征值

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
>解线性方程组（关注概念：零空间）：
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



## 概率论基础

### 课程

[掌握机器学习数学基础之概率统计（重点知识）](https://zhuanlan.zhihu.com/p/30314229?utm_medium=social&utm_source=wechat_session)

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

> 

#### 边缘概率

> 

#### 离散型随机变量

#### 连续型随机变量

#### 独立性



#### 条件独立性



#### 期望



#### 方差



#### 协方差

> $\vec Z$的协方差：$Cov(\vec Z) = E[(Z-E[Z])(Z-E[Z])^T]=E[ZZ^T]-(E[Z])(E[Z])^T$
>
> $\vec Z$：有值向量
>
> $E[\vec Z]$：$\vec Z$的期望



#### 均方误差（MSE）



#### 相关系数



#### 概率分布-离散（0-1分布）

> 

#### 概率分布-离散（几何分布）

> 

#### 概率分布-离散（二项分布）

> 

#### 概率分布-离散（泊松分布）

> 泊松分布：描述某段时间内，事件具体的发生概率。即：单位时间内独立事件发生次数的概率分布。
>
> $P(N(t)=n)=\frac{(\gamma t)^ne^{-\gamma t}}{n!}$
>
> ![img](http://www.ruanyifeng.com/blogimg/asset/2015/bg2015061010.gif)
>
> N：某种函数关系
>
> t：时间
>
> n：数量
>
> $\gamma$：事件的概率
>
> > 已知：某医院平均每小时出生3个婴儿。
> >
> > 求：2小时内1个婴儿都不出生的概率。
> >
> > 答：$P(N(2)=0)=\frac{(3*2)^0e^{-3*2}}{0!}\approx0.0025$

#### 概率分布-连续（指数分布）

> 泊松分布 => 指数分布
>
> 指数分布：事件的时间间隔的概率。即：独立事件的时间间隔的概率分布。
>
> 事件发生在时间t以后的概率是：$P(X>t)=P(N(t)=0)=\frac{(\gamma t)^0e^{-\gamma t}}{0!}=e^{-\gamma t}$
>
> 事件发生在事件t以内的概率是：$P(X\leq t)=1-P(X>t)=1-e^{-\gamma t}$
>
> ![img](http://www.ruanyifeng.com/blogimg/asset/2015/bg2015061006.gif)
>
> 重要特征：无记忆性。
>
> > 已知：某医院平均每小时出生3个婴儿。
> >
> > 求：接下来15分钟内有婴儿出生的概率。
> >
> > 答：$P(X\leq 0.25)=1-e^-3*0.25\approx0.5276$

#### 概率分布-连续（均匀分布）

> 

#### 概率分布-连续（高斯分布／正态分布）

> 

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

> 从集合角度理解：https://www.zhihu.com/question/51448623

#### 中心极限定理

> 中心极限定理：大量相互独立的随机变量，其均值的分布以正态分布为**极限**。

#### 极大似然估计

> 极大似然估计：利用已知样本结果，反推最有可能（最大概率）导致这样结果的参数值。
>
> 解释：已知样本既然已经发生，则此组样本的概率乘积应该是最大的。
>
> > 已知：
> >
> > 1、一个袋子里放有黑球，白球
> >
> > 2、每次取出一个球，得知黑白后放回袋子
> >
> > 3、取10次，结果：黑球8次，白球2次
> >
> > 问：袋子里的黑球白球比例最有可能是多少？
> >
> > 答：设P(球=黑)=p，则对应该结果的概率为$P(黑=8) =p^8(1-p)^2$
> >
> > 使用极大似然估计思考：8黑2白的结果既然已经发生，说明这个结果是所有可能结果（包括已发生和未发生的结果）中概率最大的，所以使得$P(黑=8)$取到最大值的p即为所求。
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

#### 自然对数底数e

 [数学里的 e 为什么叫做自然底数？是不是自然界里什么东西恰好是 e？ - 知乎](https://www.zhihu.com/question/20296247)



## 机器学习基础

[斯坦福大学公开课 ：机器学习课程](http://open.163.com/special/opencourse/machinelearning.html)

[Kivy-CN/Stanford-CS-229-CN: A Chinese Translation of Stanford CS229 notes 斯坦福机器学习CS229课程讲义的中文翻译](https://github.com/Kivy-CN/Stanford-CS-229-CN)

**从零开始深度学习书籍《神经网络和深度学习》推荐：**[神经⽹络与深度学习 (1).pdf_免费高速下载|百度网盘-分享无限制](https://pan.baidu.com/s/1mi8YVri)，密码：e7do

**在线版《神经网络和深度学习》**：[第一章: 利用神经网络识别手写数字 | tensorfly](http://www.tensorfly.cn/home/?p=80)



### 机器学习三要素

$\begin{cases}算法\\模型\\策略\end{cases}$

### 回归

[(1 条消息)为什么线性回归叫“回归”？ - 知乎](https://www.zhihu.com/question/47455422)

$\begin{cases}梯度下降-一阶收敛：用平面来逼近局部\\牛顿法-二阶收敛：用曲面来逼近局部\end{cases}$

### 代价函数

$\begin{cases}二次代价函数\\交叉熵代价函数\\对数似然代价函数\end{cases}$

#### 代价函数-二次代价函数



#### 代价函数-交叉熵代价函数



#### 代价函数-对数似然代价函数

$C=-ln(a_y^L)$

### 算法-梯度下降（GD／SGD）

$梯度下降\begin{cases}批量梯度下降\\增量梯度下降（IGD）／随机梯度下降（SGD）\end{cases}$

[GBDT原理详解 - ScorpioLu - 博客园](https://www.cnblogs.com/ScorpioLu/p/8296994.html)

### 算法-Adam



### 算法-线性回归

#### 求$\vec \theta$，迭代方法（慢）-最小均方算法（LMS）

$\begin{cases}线性函数：h(x)=\vec \theta^T\vec x\\成本函数：J(\theta)= \frac 12 \sum^m_{i=1}(H_\theta(x^{(i)})-y^{(i)})^2\\梯度下降：\theta_j := \theta_j - \alpha \frac \partial {\partial\theta_J}J(\theta)\end{cases}$ $\Rightarrow$ 最小均方算法（LMS）：$\theta_j := \theta_j + \alpha (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}$

#### 求$\vec \theta$，非迭代方法（快）-法线方程

法线方程：$X^TX\theta=X^T \vec y$

使得$J(\vec \theta)$最小的$\theta=(X^TX)^-1X^T\vec y$

#### 最小二乘法得到的$\theta$和最大似然法得到的$\theta$是一致的

### 算法-逻辑回归（LR-Logistic Regression、二分类）

#### 逻辑函数／双弯曲S型函数（sigmoid）（可用作激活函数、非线性）

 $h_\theta(x) = g(\theta^T x) = \frac  1{1+e^{-\theta^Tx}}$

sigmoid：$g(z)= \frac 1 {1+e^{-z}}$

$\begin{aligned}g'(z) = g(z)(1-g(z))\\\end{aligned}$

[【机器学习】Logistic Regression 的前世今生（理论篇） - CSDN博客](https://blog.csdn.net/cyh_24/article/details/50359055)

#### 求$\vec \theta$，慢，梯度上升

$\begin{cases}逻辑函数：g(z)= \frac 1 {1+e^{-z}} \\ 似然函数：\begin{aligned}L(\theta) &= p(\vec{y}| X; \theta)\\&= \prod^m_{i=1}  p(y^{(i)}| x^{(i)}; \theta)\\&= \prod^m_{i=1} (h_\theta (x^{(i)}))^{y^{(i)}}(1-h_\theta (x^{(i)}))^{1-y^{(i)}} \\\end{aligned} \\ 梯度上升：\begin{aligned}\frac  {\partial}{\partial \theta_j} l(\theta) &=(y\times \frac  1 {g(\theta ^T x)}  - (1-y)\times\frac  1 {1- g(\theta ^T x)}   )\frac  {\partial}{\partial \theta_j}g(\theta ^Tx) \\&= (y\times \frac  1 {g(\theta ^T x)}  - (1-y)\times \frac  1 {1- g(\theta ^T x)}   )  g(\theta^Tx)(1-g(\theta^Tx)) \frac  {\partial}{\partial \theta_j}\theta ^Tx \\&= (y(1-g(\theta^Tx) ) -(1-y) g(\theta^Tx)) x_j\\&= (y-h_\theta(x))x_j\end{aligned}\end{cases}$$\Rightarrow$ $\theta_j := \theta_j + \alpha (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}$

#### 求$\vec \theta$，快，牛顿法（1维）／牛顿-拉普森法（n维）

$\begin{cases}逻辑函数：g(z)= \frac 1 {1+e^{-z}} \\ 似然函数：\begin{aligned}L(\theta) &= p(\vec{y}| X; \theta)\\&= \prod^m_{i=1}  p(y^{(i)}| x^{(i)}; \theta)\\&= \prod^m_{i=1} (h_\theta (x^{(i)}))^{y^{(i)}}(1-h_\theta (x^{(i)}))^{1-y^{(i)}} \\\end{aligned} \\ 牛顿-拉普森法：\theta := \theta - H^{-1}\nabla_\theta l(\theta)，H_{ij}= \frac {\partial^2 l(\theta)}{\partial \theta_i \partial \theta_j}\end{cases}$$\Rightarrow$求得最大$l(\theta)$及其对应的$\theta$(Fisher评分）

### 算法-Softmax（柔性最大值）

神经网络特殊的输出层L：$a_j^L=\frac{e^{z_j^L}}{\sum_ke^{z_k^L}}$

其中：$z_k^L=\sum_kw_{jk}^La_k^{L-1}+b_j^L$

### 广义线性模型（GLM，Generalized Linear Models）

#### 指数族

指数族：可用以下描述的分布。

$ p(y;\eta) =b(y)exp(\eta^TT(y)-a(\eta))$

#### 广义线性模型特例：普通最小二乘法（OLS-Ordinary Least Square）



#### 广义线性模型特例：逻辑回归（LR-Logistic Regression）



#### 广义线性模型特例：softmax回归

softmax回归：逻辑回归（logistic回归）模型在多分类问题上的推广。

$处理分类问题\begin{cases}k个logistic回归（二分类器、基于伯努利分布）：各分类非互斥。\\softmax回归（多分类器、基于多项式分布）：各分类是互斥。\end{cases}$

softmax：可以用来给不同对象分配概率。

#### 广义线性模型特例：多项式分布（多值化）



#### 广义线性模型特例：泊松分布（Possion）



#### 广义线性模型特例：$\beta$和狄利克雷分布



#### 广义线性模型特例：$\gamma$和指数分布



#### 伯努利分布（Bernoulli分布）（二值化）



### 高斯判别分析（GDA）

#### 多元正态分布／多变量高斯分布



#### 高斯判别分析（GDA）

$\begin{cases}多元正态分布\\\begin{cases}\begin{aligned}y & \sim Bernoulli(\phi)\\x|y = 0 & \sim N(\mu_o,\Sigma)\\x|y = 1 & \sim N(\mu_1,\Sigma)\\\end{aligned}\end{cases}\end{cases}$$\Rightarrow$高斯判别分析（GDA）

**高斯判别分析**，是一种特殊的逻辑回归。所以**逻辑回归**比**高斯判别分析**更抽象，**高斯判别分析**比**逻辑回归**更精确。

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229note2f6.png)

### 朴素贝叶斯（NB）

#### 朴素贝叶斯假设

朴素贝叶斯假设：对于特定y，特征向量$X_i$是独立的。即：$p(x_i|y)=p(x_i|y, x_j)$

当原生的连续值的属性不太容易用一个**多元正态分布**来进行建模的时候，将其特征向量离散化后使用**朴素贝叶斯（NB）**来替代**高斯判别分析（GDA）**，通常能形成一个更好的**分类器**。

#### 拉普拉斯光滑（Laplace）



#### 文本分类事件模型



### 决策树

#### 回归树



#### 分类树



### 算法-GBDT（梯度提升决策树）

[GBDT详解 - 白开水加糖 - 博客园](https://www.cnblogs.com/peizhe123/p/5086128.html)



#### 线性可分

[线性可分](https://www.cnblogs.com/lgdblog/p/6858832.html)

#### 函数边界／几何边界



#### 拉格朗日（Lagrange）乘子法

拉格朗日乘子法：把目标函数和等式约束统一到拉格朗日函数中。

[拉格朗日乘子法 - 搜索结果 - 知乎](https://www.zhihu.com/search?type=content&q=%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E5%AD%90%E6%B3%95)

#### 拉格朗日对偶性



### 算法-SVM（Support Vector Machine、支持向量机）

[[推荐]SVM入门（一）至（三）Refresh - Jasper's Java Jacal - BlogJava](http://www.blogjava.net/zhenandaci/archive/2016/02/29/254519.html)

松弛变量：

惩罚因子：

倾斜问题：



[支持向量机(SVM)是什么意思？ - 知乎](https://www.zhihu.com/question/21094489)

[SVM with polynomial kernel visualization - YouTube](https://www.youtube.com/watch?v=3liCbRZPrZA)

#### 最优边界分类器



#### 核（Kernels）



#### 坐标上升算法



#### SMO



#### LR（Logistic Regression） vs SVM（Support Vector Machine）

[Logistic Regression and SVM](http://charleshm.github.io/2016/03/LR-SVM/?utm_medium=social&utm_source=wechat_session#fn:1)



### xgboost



### 准确率／召回率

**准确率**：针对测试结果（分母），样本中的正例有多少被预测正确。

**召回率**：针对样本（分母），预测为正的样本中有多少是真正的正样本。

### 熵

**熵**的本质：香农信息量的期望。

$香农信息量 = log_2\frac{1}{p}$

#### 信息熵

**信息熵**：随机变量或整个系统的不确定性。

$信息熵 = \sum^n_{k=1}p_klog_2\frac{1}{p_k}$

$p_k​$：真实分布

信息熵：信息熵是消除系统的不确定性的最小代价。

[如何通俗的解释交叉熵与相对熵? - 知乎](https://www.zhihu.com/question/41252833/answer/378143666)

#### 交叉熵

**交叉熵**：用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小。

$交叉熵 = \sum^n_{k=1}p_klog_2\frac{1}{q_k}$

$p_k$：真实分布

$q_k$：非真实分布

#### 相对熵

**相对熵**：用来衡量两个取值为正的函数或概率分布之间的差异。

$KL(f(x) || g(x)) = \sum_{x\in X}f(x)log_2\frac{f(x)}{g(x)}$



### 感知机

**非常推荐：**[[Machine Learning & Algorithm] 神经网络基础 - Poll的笔记 - 博客园](http://www.cnblogs.com/maybe2030/p/5597716.html#_label0)

![img](https://images2015.cnblogs.com/blog/764050/201606/764050-20160619111613406-1210494225.png)

![img](https://images2015.cnblogs.com/blog/764050/201606/764050-20160619145050085-1140057304.jpg)

#### 最简单的非线性可分问题：异或问题

### 算法-反向传播算法（BP）

#### 反向传播算法

参考书籍（第34页）[神经⽹络与深度学习 (1).pdf_免费高速下载|百度网盘-分享无限制](https://pan.baidu.com/s/1mi8YVri)，密码：e7do

#### 梯度消失问题／梯度激增问题

[[Machine Learning] 深度学习中消失的梯度 - Poll的笔记 - 博客园](http://www.cnblogs.com/maybe2030/p/6336896.html)

神经网络可以计算任何函数的可视化证明，参考书籍（第137页）[神经⽹络与深度学习 (1).pdf_免费高速下载|百度网盘-分享无限制](https://pan.baidu.com/s/1mi8YVri)，密码：e7do



### 激活函数

![img](https://img-blog.csdn.net/20161031024815093?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### 激活函数-sigmoid（S型神经元）

sigmoid：$g(z)= \frac 1 {1+e^{-z}}$

### 激活函数-tanh（tanch神经元、双曲正切函数、hyperbolic tangent）

tanch：$tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

由于\sigma(z)=\frac{1+tanh(z/2)}{2}，所以tanch是sigmoid函数按比例变化的版本。

### 激活函数-ReLU（Rectified linear unit）（修正线性神经元、修正线性单元）

$f(x) = max(0, x)$

$\begin{cases} Leaky\ ReLU\\Parametric\ ReLU \\Randomized\ ReLU\\Noisy\ ReLU\end{cases}$

[神经网络回顾-Relu激活函数 - 1357 - 博客园](http://www.cnblogs.com/qw12/p/6294430.html)

[线性整流函数_百度百科](https://wapbaike.baidu.com/item/%E7%BA%BF%E6%80%A7%E6%95%B4%E6%B5%81%E5%87%BD%E6%95%B0/20263760?fr=aladdin)

### 神经网络

$$原始输入空间\rightarrow \begin{cases}矩阵线性变换\\激活函数非线形变换\end{cases}\rightarrow 线性可分／稀疏空间\rightarrow\begin{cases}分类\\回归\end{cases}$$

$\begin{cases}增加节点数\rightarrow增加维度\rightarrow增加线性转换能力\\增加层数\rightarrow增加激活函数次数\rightarrow增加非线性转换次数\end{cases}$

### 神经网络可逼近任意连续函数

#### 阶跃函数

[【神经网络和深度学习】笔记 - 第四章 神经网络可以实现任意函数的直观解释 - 野路子程序员 - 博客园](https://www.cnblogs.com/yeluzi/p/7491619.html)



### 神经网络可以计算任何函数的可视化证明

参考书籍（第111页）[神经⽹络与深度学习 (1).pdf_免费高速下载|百度网盘-分享无限制](https://pan.baidu.com/s/1mi8YVri)，密码：e7do

### 反向传播算法（BP-Back Propagation）

#### 神经元上的误差

$第l层底j个神经元的带权输入：z_j^l$

$第l层的第j个神经元的误差：\delta_j^l=\frac{\partial C}{\partial z_j^l}$

$C：代价函数$

$z_j^l=\sum_kw_{jk}^la_k^{l-1}+b_j^l\rightarrow z^l=w^la^{l-1}+b^l$

$a_j^l=\sigma(z_j^l)\rightarrow a^l=\sigma(z^l)：$

#### 反向传播的四个基本方程

$\begin{cases}输出误差：\delta^L=\nabla_aC\bigodot\sigma'(z^L)\\反向传播：\delta^l=((w^{l+1})^T\delta^{l+1})\bigodot\sigma'(z^l)\\偏置梯度：\frac{\partial C}{\partial b_j^{l}}=\delta_j^{l}\\权重梯度：\frac{\partial C}{\partial w_{jk}^{l}}=a_k^{l-1}\delta_j^l\end{cases}$

参考书籍（第41页）[神经⽹络与深度学习 (1).pdf_免费高速下载|百度网盘-分享无限制](https://pan.baidu.com/s/1mi8YVri)，密码：e7do

#### 反向传播算法步骤

$\begin{cases}1、拆训练集：将训练集分为多批\vec x，每批m个\vec x，针对每个\vec x进行以下计算\\2、前向传播：对于l=2,3,…,L，计算z^{x,l}=w^la^{x,l-1}+b，a^{x,l}=\sigma(z^{x,l})\\3、输出误差：\delta^{x,L}=\nabla_aC\bigodot\sigma'(z^{x,L})\\4、反向传播：对于l=L-1,L-2,…,2，计算\delta^{x,l}=((w^{l+1})^T\delta^{x,l+1})\bigodot\sigma'(z^{x,l})\\5、梯度下降：对于l=L-1,L-2,...2，更新w^l=w^l-\frac{\eta}{m}\sum_x\delta^{x,l}(a^{x,l-1})^T，b^l=b^l-\frac{\eta}{m}\sum_x\delta^{x,l}\end{cases}$

### 拟合

$\begin{cases}欠拟合\rightarrow增加特征量／完善模型\\过拟合\rightarrow减少特征量／正则化\end{cases}$

$减轻过拟合\begin{cases}正则化／规范化-L1：\\正则化／规范化-L2：\\弃权-Dropout：确保模型对丢失某些个体连接的场景更加健壮。\\人为增加训练样本：\end{cases}$

### 正则化／规范化

正则化／规范化：寻找**小的权重**和**最小化原始代价函数**之间的折中，由参数$\lambda$控制。

更小的权重意味着网络的行为不会因为一个输入的改变而变化太大，即学习局部噪声的影响更加困难，即抵抗噪声能力强。

#### L1／L2

$\begin{cases}L1：C=C_0+\frac{\lambda}{2n}\sum_w|w|\\L2：C=C_0+\frac{\lambda}{2n}\sum_ww^2\end{cases}$

C_0：原始的代价函数

（）\lambda（\lambda>0）：规范化参数

#### 梯度下降-L1

L2之后，梯度下降的权重学习规则变成：w=w-\frac{\eta\lambda}{n}sgn(w)-\eta\frac{\partial C_0}{\partial w}

L2之后，梯度下降的偏置学习规则变成：$b=b-\eta\frac{\partial C_0}{\partial b}$

$sgn(w)：w的正负符号$

$sgn(0)=0$

#### 梯度下降-L2

L2之后，梯度下降的权重学习规则变成：w=(1-\frac{\eta\lambda}{n})w-\eta\frac{\partial C_0}{\partial w}

L2之后，梯度下降的偏置学习规则仍是：$b=b-\eta\frac{\partial C_0}{\partial b}$

#### L1 vs L2

$\begin{cases}L1\begin{cases}|w|较大时：不敏感\\|w|较小时：敏感\end{cases}\\L2\begin{cases}|w|较大时：敏感\\|w|较小时：不敏感\end{cases}\end{cases}$

L1规范化倾向于聚集网络的权重在相对少量的高重要度连接上，而其他权重就会被驱使向0接近。

### Dropout



### 权重初始化



### 超参数调整



### 策略-提前停止（Early Stopping）



### 算法-Hessian技术／Hessian优化

？

### 算法-基于momentum的梯度下降

？



### CNN-卷积神经网络

#### 卷积层（Conv Layers）

##### 感受野 -> 特征映射（Feature Map）



##### 卷积核／滤波器



##### 共享权重／共享偏置



#### 混合层／池化层（Pooling Layers）

$\begin{cases}最大值混合（max-pooling）：对领域特征点取最大值\\平均池化（mean-pooling）：对领域特征点求平均\\L2混合（L2-pooling）：对领域特征点取平方和之后算平方根\end{cases}$

##### max-pooling

##### mean-pooling

##### L2-pooling



#### 全连接层



### RNN-循环神经网络

RNN在处理时序数据和过程上效果特别不错。如语音识别、自然语言处理等。



### DNN-深度神经网络



### LSTMs-长短期记忆单元（Long short-term memory units）



### DBN-深度信念网络（生成式模型和Boltzmann机）



### Conway法则



### 泰勒展开式

[(2 条消息)怎样更好地理解并记忆泰勒展开式？ - 知乎](https://www.zhihu.com/question/25627482/answer/313088784)



## 斯坦福课程学习

### 课程

斯坦福大学公开课 ：机器学习课程](http://open.163.com/special/opencourse/machinelearning.html)

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
	
支持向量机-SVM



* 回归问题
* 分类问题


[Octave_for_macOS下载 ](http://wiki.octave.org/Octave_for_macOS)

---

### 笔记：[第2课]监督学习应用.梯度下降


线性回归
	m：训练集大小
	x：输入
	y：输出
	(x, y)：一个样本

梯度下降算法
	局部最优问题
	批梯度下降算法
	随机梯度下降算法／增量梯度下降算法
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

### 笔记：[第5课]

### 笔记：[第6课]

### 笔记：[第7课]

### 笔记：[第8课]

### 笔记：[第9课]

### 笔记：[第10课]

### 笔记：[第11课]

### 笔记：[第12课]



## Python

#### MNIST by Python

**从零开始深度学习书籍推荐：**[神经⽹络与深度学习 (1).pdf_免费高速下载|百度网盘-分享无限制](https://pan.baidu.com/s/1mi8YVri)，密码：e7do

#### API

np.linalg.inv()：矩阵求逆

np.linalg.det()：矩阵求行列式

np.linalg.norm()：



## Tensorflow

### 概念

#### Graph：图，使用图来表示计算任务。

#### Session：上下文，图在此上下文中启动执行。

#### Tensor：张量，表示数据。

#### Variable：变量，维护状态。

#### Feed/Fetch：为任何操作（Operation，节点）写数据／读数据

#### Shape：



#### 检查点文件

#### 事件文件



### API

* tf.Variable() : 变量

* tf.constant() : 常量

* tf.placeholder(） : 占位符

* tf.add() : 加

* tf.assign(): 赋值

* tf.log() :

* tf.mul() :

* tf.matmul() : 矩阵乘法

* tf.reduce_sum() : 计算元素和

* tf.argmax() : 获取向量最大值的索引

* tf.cast() ： 映射到指定类型

* tf.equal() :

* tf.reduce_mean() : [张量不同数轴的平均值计算](https://www.cnblogs.com/yuzhuwei/p/6986171.html)

* tf.truncated_normal(shape, mean, stddev) ：产生满足正太分布的随机数（shape-张量维度，mean-均值，stddev-标准差），产生的随机数与均值的差距不会超过两倍的标准差。

  [tf.truncated_normal的用法 - CSDN博客](https://blog.csdn.net/uestc_c2_403/article/details/72235565)

* tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)：卷积函数 [【TensorFlow】tf.nn.conv2d是怎样实现卷积的？ - CSDN博客](https://blog.csdn.net/mao_xiao_feng/article/details/53444333)

* tf.nn.softmax() : softmax模型

* sess = tf.Session() : 启动会话

* sess.run() : 执行图

* sess.close() : 关闭会话




### TensorFlow安装

#### Mac平台

```
1、安装Anaconda
2、通过conda建立Tensorflow运行环境
3、激活Tensorflow运行环境
4、安装Pycharm IDE
```

[Mac下TensorFlow安装及环境搭建](https://www.cnblogs.com/vijozsoft/p/7832229.html)

[Anaconda下载](https://www.anaconda.com/download)

[TensorFlow中文社区](http://www.tensorfly.cn/)

#### Linux（CentOS6.9）

pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

错误处理：[InsecurePlatformWarning: A true SSLContext object is not available. This prevents urllib3 from configuring SSL appropriately [duplicate]](https://stackoverflow.com/questions/29134512/insecureplatformwarning-a-true-sslcontext-object-is-not-available-this-prevent)

### TensorFlow使用

```
激活Tensorflow环境：
source activate tensorflow
退出Tensorflow环境：
source deactivate tensorflow
```


### TensorFlow学习

http://blog.csdn.net/shingle_/article/details/52653621



### TensorBoard

[详解 TensorBoard－如何调参 - 简书](https://www.jianshu.com/p/d059ffea9ec0)
[TensorBoard--TensorFlow可视化](http://blog.csdn.net/wangjian1204/article/details/53291619)


#### tensorboard --logdir

tensorboard --logdir=./tmp/mnist

注意："="左右边不可以有空格

或

tensorboard —logdir ./tmp/mnist



### Tensorflow-MNIST

[MNIST机器学习入门](http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html)

[深层学习为何要“Deep”（上）](https://zhuanlan.zhihu.com/p/22888385)



## Theano





## PyTorch





## Keras

[[DSC 2016] 系列活動：李宏毅 / 一天搞懂深度學習](https://www.slideshare.net/tw_dsconf/ss-62245351?qid=108adce3-2c3d-4758-a830-95d0a57e46bc&v=&b=&from_search=3)

#### Maxout



