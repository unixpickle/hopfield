"""
Test Hebbian training.
"""

import unittest

import numpy as np
import tensorflow as tf

from .hebb import hebbian_update

# pylint: disable=E1129

class TestHebb(unittest.TestCase):
    """
    Tests for Hebbian learning.
    """
    def test_batch_consistency(self):
        """
        Test that the batch size during training does not
        affect the end result.
        """
        with tf.Graph().as_default():
            with tf.Session() as sess:
                mat_1 = tf.Variable(tf.zeros((20, 20), dtype=tf.float64))
                mat_2 = tf.Variable(tf.zeros((20, 20), dtype=tf.float64))
                data = np.random.randint(0, 2, size=(15, 20), dtype='bool')
                data_ph = tf.placeholder(tf.bool, shape=(None, 20))
                update_1 = hebbian_update(data_ph, mat_1)
                update_2 = hebbian_update(data_ph, mat_2)
                sess.run(tf.global_variables_initializer())

                sess.run(update_1, feed_dict={data_ph: data})

                sess.run(update_2, feed_dict={data_ph: data[:3]})
                for row in data[3:]:
                    sess.run(update_2, feed_dict={data_ph: [row]})

                actual = _hebbs_rule(data)
                self.assertTrue(np.allclose(sess.run(mat_1), actual))
                self.assertTrue(np.allclose(sess.run(mat_2), actual))

def _hebbs_rule(data):
    signed_data = data.astype('float64')*2 - 1
    res = np.matmul(np.transpose(signed_data), signed_data) / len(data)
    for i in range(len(res)):
        res[i, i] = 0
    return res
