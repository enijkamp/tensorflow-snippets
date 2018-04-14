import tensorflow as tf


@tf.RegisterGradient("CustomClipGrad")
def _clip_grad(unused_op, grad):
    return tf.clip_by_value(grad, -0.1, 0.1)


with tf.Graph().as_default() as g:

    input = tf.Variable([3.0], dtype=tf.float32)

    with g.gradient_override_map({"Identity": "CustomClipGrad"}):
        output_clip = tf.identity(input, name="Identity")
    grad_clip = tf.gradients(output_clip, input)

    output = tf.identity(input)
    grad = tf.gradients(output, input)

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    print("with clipping:", sess.run(grad_clip)[0])
    print("without clipping:", sess.run(grad)[0])
