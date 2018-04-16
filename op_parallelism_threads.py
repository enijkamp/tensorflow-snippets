import tensorflow as tf

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

with tf.Session() as sess:

    x = tf.placeholder(tf.int32, shape=[4])
    y = tf.Print(x, [tf.shape(x)], message='tf.shape(x)=', summarize=4)
    y.eval(feed_dict={x: [0, 1, 2, 3]})
