import tensorflow as tf

# https://blog.csdn.net/lengguoxing/article/details/78456279
# TensorFlow的数据中央控制单元是tensor(张量)，一个tensor由一系列的原始值组成，这些值被形成一个任意维数的数组。一个tensor的列就是它的维度。
# Building the computational graph构建计算图
# 一个computational graph(计算图)是一系列的TensorFlow操作排列成一个节点图
node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

print('*'*50)
# 一个session封装了TensorFlow运行时的控制和状态,要得到最终结果要用session控制
session = tf.Session()
print(session.run([node1,node2]))

print('*'*50)
# 组合Tensor节点操作(操作仍然是一个节点)来构造更加复杂的计算
node3 = tf.add(node1,node2)
print(node3)
print(session.run(node3))

print('*'*50)
# TensorFlow提供一个统一的调用称之为TensorBoard，它能展示一个计算图的图片
# 一个计算图可以参数化的接收外部的输入，作为一个placeholders(占位符)，一个占位符是允许后面提供一个值的
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_node = a+b

print(session.run(add_node,{a:3,b:4}))
print(session.run(add_node,{a:[1,4],b:[4,9]}))
print('*'*50)

# 我们可以增加另外的操作来让计算图更加复杂
add_and_triple = add_node * 3
print(session.run(add_and_triple,{a:3,b:4}))

print('*'*50)
# 构造线性模型输入 y = w*x+b
w = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
linner_mode = w*x + b

# 当你调用tf.constant时常量被初始化，它们的值是不可以改变的，而变量当你调用tf.Variable时没有被初始化，在TensorFlow程序中要想初始化这些变量，你必须明确调用一个特定的操作
init = tf.global_variables_initializer()
session.run(init)
print(session.run(linner_mode,{x:[1,2,3,4,8]}))

print('*'*50)
# 评估模型好坏，我们需要一个y占位符来提供一个期望值，和一个损失函数
y = tf.placeholder(dtype=tf.float32)
loss_function = tf.reduce_sum(tf.square(linner_mode - y))
print(session.run(loss_function,{x:[1,2,3,4],y:[0, -1, -2, -3]}))

# 们分配一个值给W和b(得到一个完美的值是-1和1)来手动改进这一点,一个变量被初始化一个值会调用tf.Variable，但是可以用tf.assign来改变这个值，例如：fixW = tf.assign(W, [-1.])
fixw = tf.assign(w,[-1])
fixb = tf.assign(b,[1])
session.run([fixb,fixw])
print(session.run(loss_function,{x:[1,2,3,4],y:[0, -1, -2, -3]}))

# optimizers 我们写一个优化器使得，他能慢慢改变变量来最小化损失函数，最简单的是梯度下降
optimizers = tf.train.GradientDescentOptimizer(0.01)
train = optimizers.minimize(loss_function)
session.run(init) # reset value
for i in range(1000):
    session.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print(session.run([w,b]))
print(session.run([w,b]))