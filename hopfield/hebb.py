"""
Train Hopfield networks with Hebb's rule.
"""

import tensorflow as tf

def hebbian_update(samples, weights):
    """
    Create an Op that updates the weight matrix with the
    mini-batch via the Hebbian update rule.

    Args:
      samples: a mini-batch of samples. Should be a 2-D
        Tensor with a dtype of tf.bool.
      weights: the weight matrix to update. Should start
        out as all zeros.

    Returns:
      An Op that updates the weights such that the batch
        of samples is encouraged.

    Hebbian learning involves a running average over all
    of the training data. This is implemented via extra
    training-specific variables.
    """
    assert samples.dtype == tf.bool
    assert len(samples.get_shape()) == 2

    dtype = weights.dtype

    # Maintain an unbiased running average.
    counter = tf.get_variable(weights.op.name + '/hebb_counter',
                              shape=(),
                              dtype=tf.int32,
                              initializer=tf.zeros_initializer(),
                              trainable=False)
    old_count = tf.cast(counter, dtype)
    new_count = tf.cast(tf.shape(samples)[0], dtype)
    rate = new_count / (new_count + old_count)

    numerics = 2*tf.cast(samples, dtype) - 1
    outer = tf.matmul(tf.transpose(numerics), numerics) / new_count

    with tf.control_dependencies([outer]):
        return tf.group(tf.assign_add(weights, rate*(outer-weights)),
                        tf.assign_add(counter, tf.shape(samples)[0]),
                        name='hebb')
