"""
Update rules for Hopfield networks.
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
    return _second_moment_update(samples, weights)[0]

def covariance_update(samples, weights, thresholds):
    """
    Create an Op that performs a statistically centered
    Hebbian update on the mini-batch.

    This is like hebbian_update(), except that the weights
    are trained on a zero-mean version of the samples, and
    the thresholds are tuned as well as the weights.
    """
    dtype = weights.dtype
    second_moment = tf.get_variable(weights.op.name + '/second_moment',
                                    shape=weights.get_shape(),
                                    dtype=dtype,
                                    initializer=tf.zeros_initializer(),
                                    trainable=False)
    update_second, new_second, rate = _second_moment_update(samples, second_moment,
                                                            mask_diag=False)
    new_mean = tf.negative(tf.reduce_mean(tf.cast(samples, dtype=dtype)*2-1, axis=0))
    new_thresh = tf.assign_add(thresholds, rate * (new_mean - thresholds))
    outer = tf.matmul(tf.expand_dims(new_thresh, axis=1),
                      tf.expand_dims(new_thresh, axis=0))
    return tf.group(update_second, tf.assign(weights, new_second - outer))

def extended_storkey_update(sample, weights):
    """
    Create an Op that performs a step of the Extended
    Storkey Learning Rule.

    Args:
      sample: a 1-D sample Tensor of dtype tf.bool.
      weights: the weight matrix to update.

    Returns:
      An Op that updates the weights based on the sample.
    """
    scale = 1 / int(weights.get_shape()[0])
    numerics = 2*tf.cast(sample, weights.dtype) - 1
    row_sample = tf.expand_dims(numerics, axis=0)
    row_h = tf.matmul(row_sample, weights)

    pos_term = (tf.matmul(tf.transpose(row_sample), row_sample) +
                tf.matmul(tf.transpose(row_h), row_h))
    neg_term = (tf.matmul(tf.transpose(row_sample), row_h) +
                tf.matmul(tf.transpose(row_h), row_sample))
    return tf.assign_add(weights, scale * (pos_term - neg_term))

def _second_moment_update(samples, weights, mask_diag=True):
    """
    Get an Op to do an uncentered second-moment update.

    Returns (update_op, updated_weights, running_avg_rate)
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

    if not mask_diag:
        rate_mask = rate
    else:
        diag_mask = 1 - tf.diag(tf.ones((tf.shape(weights)[0],), dtype=dtype))
        rate_mask = rate * diag_mask

    with tf.control_dependencies([outer, rate_mask]):
        new_weights = tf.assign_add(weights, rate_mask*(outer-weights))
        return tf.group(new_weights,
                        tf.assign_add(counter, tf.shape(samples)[0]),
                        name='hebb'), new_weights, rate
