# eof_tensorflow

# 概念

* Graph：图，使用图来表示计算任务。
* Session：上下文，图在此上下文中启动执行。
* Tensor：张量，表示数据。
* Variable：变量，维护状态。
* Feed/Fetch：为任何操作（Operation，节点）写数据／读数据

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
* tf.equal() :
* tf.reduce_mean() : [张量不同数轴的平均值计算](https://www.cnblogs.com/yuzhuwei/p/6986171.html)
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
