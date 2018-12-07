import tensorflow as tf

a = tf.Variable([[[1, 1, 1,1], [2, 2, 2,1]], [[3, 3, 3,1], [4, 4, 4,1]], [[3, 3, 3,1], [4, 4, 4,1]]],dtype=tf.int32)
print(tf.Session().run(tf.shape(a,name='aaa')))
print(tf.Session().run(tf.size(a,name='aaa')))
b = tf.Variable([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
print(tf.Session().run(tf.rank(a)))
# init = tf.global_variables_initializer()
# print(tf.Session().run(tf.reshape(b,[-1,1])))