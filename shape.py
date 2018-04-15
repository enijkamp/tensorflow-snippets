import tensorflow as tf

with tf.Session() as sess:

    x = tf.placeholder(tf.int32, shape=[4])
    print(x.get_shape()) # (4,)

    y, _ = tf.unique(x)
    print(y.get_shape()) # (?,)
    print(y.eval(feed_dict={x: [0, 1, 2, 3]}).shape) # (4,)
    print(y.eval(feed_dict={x: [0, 0, 0, 0]}).shape) # (1,)

    z = tf.shape(y)
    print(z.eval(feed_dict={x: [0, 1, 2, 3]})) # [4]
    print(z.eval(feed_dict={x: [0, 0, 0, 0]})) # [1]
