import tensorflow as tf

cte = tf.constant('hello, world!')

x = tf.placeholder(tf.int16)
y = tf.placeholder(tf.int16)

with tf.Session() as sess:
    print(sess.run(cte))
    print(sess.run(x + y, feed_dict={x: 1, y: 2}))


