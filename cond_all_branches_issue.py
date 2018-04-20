import tensorflow as tf

def expected():
    x1 = tf.Variable(1.)
    x2 = tf.Variable(1.)

    assign_1 = tf.assign(x1, 2.)
    assign_2 = tf.assign(x2, 2.)

    with tf.control_dependencies([assign_1]):
        result_1 = tf.identity(x1)
    with tf.control_dependencies([assign_2]):
        result_2 = tf.identity(x1)

    cond_result = tf.cond(tf.Variable(True), lambda: result_1, lambda: result_2)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(cond_result)
        print(sess.run([x1, x2]))

def corrected():
    x1 = tf.Variable(1.)
    x2 = tf.Variable(1.)

    def f_true():
        assign_1 = tf.assign(x1, 2.)
        with tf.control_dependencies([assign_1]):
            return tf.identity(x1)
    def f_false():
        assign_2 = tf.assign(x2, 2.)
        with tf.control_dependencies([assign_2]):
            return tf.identity(x2)

    cond_result = tf.cond(tf.Variable(True), f_true, f_false)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(cond_result)
        print(sess.run([x1, x2]))

def simplified():
    x1 = tf.Variable(1.)
    x2 = tf.Variable(1.)

    cond_result = tf.cond(tf.Variable(True), lambda: tf.assign(x1, 2.), lambda: tf.assign(x2, 2.))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(cond_result)
        print(sess.run([x1, x2]))


expected()    # [2.0, 2.0]
corrected()   # [2.0, 1.0]
simplified()  # [2.0, 1.0]
