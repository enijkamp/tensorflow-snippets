import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

with tf.Session(config=config) as sess:

    x = tf.placeholder(tf.int32, shape=[4])
    y = tf.Print(x, [x], message='x=', summarize=4)
    y.eval(feed_dict={x: [0, 1, 2, 3]})
