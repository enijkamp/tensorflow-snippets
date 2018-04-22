import sys
import os
import logging
from datetime import datetime
from typing import NamedTuple

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def make_dir(dir):
    if not tf.gfile.Exists(dir):
        tf.gfile.MakeDirs(dir)
    return dir

def initialize_logging(file_name):
    log_format = logging.Formatter('%(asctime)s : %(message)s')
    logger = logging.getLogger()
    logger.handlers = []

    logger_tf = logging.getLogger('tensorflow')
    logger_tf.setLevel(logging.INFO)
    logger_tf.handlers = []

    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

def count_params(vars):
    total_parameters = 0
    for var in vars:
        variable_parameters = 1
        for dim in var.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def get_current_lr(adam):
    _beta1_power, _beta2_power = adam._get_beta_accumulators()
    current_lr = (adam._lr_t * tf.sqrt(1 - _beta2_power) / (1 - _beta1_power))
    return current_lr

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def model(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 16])
    b_conv2 = bias_variable([16])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 16, 8])
    b_fc1 = bias_variable([8])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([8, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv

class Config(NamedTuple):
    seed: int = 1
    max_steps: int = 200
    learning_rate: float = 1e-4
    batch_size: int = 128
    log_interval: int = 100

def train(config, data, output_dir):
    tf.logging.info('config={0}'.format(config))

    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        y_conv = model(x)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        adam = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        train_step = adam.minimize(cross_entropy)
        lr = get_current_lr(adam)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        initializer = tf.global_variables_initializer()

        saver = tf.train.Saver()

        graph.finalize()

        tf.logging.info('size(parameters)={0}'.format(count_params(tf.trainable_variables())))

    with tf.Session(graph=graph).as_default() as sess:
        tf.logging.info('training')
        initializer.run()
        for i in range(config.max_steps):
            batch = data.train.next_batch(config.batch_size)
            if i % config.log_interval == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                tf.logging.info('{0:5d} -> accuracy(training)={1:.5f} lr={2:.5f}'.format(i, train_accuracy, lr.eval()))
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        tf.logging.info('accuracy(testing)={0.5f}'.format(accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels})))

        saver.save(sess, output_dir)

if __name__ == '__main__':
    config = Config()
    tf.set_random_seed(seed=config.seed)

    run_name = datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    output_dir = make_dir(os.path.join('output', run_name))
    data_dir = make_dir(os.path.join('data', 'mnist'))

    initialize_logging(os.path.join(output_dir, 'log.txt'))

    train(config, input_data.read_data_sets(data_dir, one_hot=True), os.path.join(output_dir, 'model'))
