# eof_tensorflow

# 深度学习

## 概念

*   ​

[深层学习为何要“Deep”（上）](https://zhuanlan.zhihu.com/p/22888385)



## 斯坦福课程学习

[斯坦福大学公开课 ：机器学习课程](http://open.163.com/special/opencourse/machinelearning.html)

### 笔记：[第1课]机器学习的动机与应用

```
Matlab / Octave：课程作业完成工具
Matlab - 收费
Octave - 免费
机器学习定义
	任务T
	评估P
	经验E
	
课程4大部份：
	监督学习：有标准答案
		分类问题（分类算法）
	学习理论：
	无监督学习：无标准答案（从数据中挖掘有趣的结构）
		聚类问题（聚类算法）
			如“鸡尾酒会问题”-独立组件分析
	强化学习：
		回报函数
	
支持向量机
```

[Octave_for_macOS下载 ](http://wiki.octave.org/Octave_for_macOS)



### 笔记：[第2课]监督学习应用.梯度下降

```
线性回归（一种参数学习算法）
	m：训练集大小
	x：输入
	y：输出
	(x, y)：一个样本
	
梯度下降
	局部最优问题
	批梯度下降算法
	随机梯度下降算法/增量梯度下降算法
	
	矩阵的迹：n×n矩阵A的主对角线（从左上方至右下方的对角线）上各个元素的总和。
	矩阵转置：
	
正规方程组
	正规方程组：（使得最小二乘拟合问题可以不再用梯度下降算法进行迭代计算）
	最小二乘拟合问题：
```



### 笔记：[第3课]欠拟合与过拟合的概念

```
欠拟合
过拟合

防止欠拟合/过拟合，使用特征选择算法

参数学习算法：有固定数目的参数，以用来进行数据拟合的算法
非参数学习算法：参数数量会随着训练集级大小m线性增长的算法。（每次预测时，都需要重新拟合一次，计算量较大。使得此算法对于大型数据集更高效的方法是：KD tree by Andrew Moore）

局部加权回归（Loess）（一种非参数学习算法）

logistic回归

感知器

牛顿方法

```



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
