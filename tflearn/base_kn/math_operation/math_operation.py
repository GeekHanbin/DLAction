from __future__ import print_function
import tensorflow as tf
import os

tf.app.flags.DEFINE_string('log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs','Directory where event logs are written to.')

FLAGS = tf.app.flags.FLAGS
if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')

a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("output: ", sess.run([a,b,x,y]))

# Closing the writer.
writer.close()
sess.close()