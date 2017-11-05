"""
Test Hebbian training.
"""

import unittest

import numpy as np
import tensorflow as tf

from .update import hebbian_update, covariance_update

# pylint: disable=E1129

class TestHebb(unittest.TestCase):
    """
    Tests for Hebbian learning.
    """
    def test_full_batch(self):
        """
        Test with a single batch.
        """
        self._test_final_weights(False)

    def test_mini_batches(self):
        """
        Test with a mini-batch.
        """
        self._test_final_weights(True)

    def _test_final_weights(self, mini_batches):
        """
        Test the final weights.
        """
        with tf.Graph().as_default():
            with tf.Session() as sess:
                mat = tf.Variable(tf.zeros((20, 20), dtype=tf.float64))
                data = np.random.randint(0, 2, size=(15, 20), dtype='bool')
                data_ph = tf.placeholder(tf.bool, shape=(None, 20))
                update = hebbian_update(data_ph, mat)
                sess.run(tf.global_variables_initializer())
                if mini_batches:
                    sess.run(update, feed_dict={data_ph: data[:3]})
                    for row in data[3:]:
                        sess.run(update, feed_dict={data_ph: [row]})
                else:
                    sess.run(update, feed_dict={data_ph: data})
                actual = _hebbs_rule(data)
                self.assertTrue(np.allclose(sess.run(mat), actual))

class TestCovariance(unittest.TestCase):
    """
    Test covariance updates.
    """
    def test_full_batch(self):
        """
        Test with a single batch.
        """
        self._test_final_weights(False)

    def test_mini_batches(self):
        """
        Test with a mini-batch.
        """
        self._test_final_weights(True)

    def _test_final_weights(self, mini_batches):
        """
        Test covariance updates.
        """
        with tf.Graph().as_default():
            with tf.Session() as sess:
                mat = tf.Variable(tf.zeros((20, 20), dtype=tf.float64))
                thresh = tf.Variable(tf.zeros((20,), dtype=tf.float64))
                data = np.random.randint(0, 2, size=(15, 20), dtype='bool')
                data_ph = tf.placeholder(tf.bool, shape=(None, 20))
                update = covariance_update(data_ph, mat, thresh)
                sess.run(tf.global_variables_initializer())
                if mini_batches:
                    sess.run(update, feed_dict={data_ph: data[:3]})
                    for row in data[3:]:
                        sess.run(update, feed_dict={data_ph: [row]})
                else:
                    sess.run(update, feed_dict={data_ph: data})
                actual_m, actual_t = _covariance_rule(data)
                self.assertTrue(np.allclose(sess.run(mat), actual_m))
                self.assertTrue(np.allclose(sess.run(thresh), actual_t))

def _hebbs_rule(data):
    signed_data = data.astype('float64')*2 - 1
    res = np.matmul(np.transpose(signed_data), signed_data) / len(data)
    for i in range(len(res)):
        res[i, i] = 0
    return res

def _covariance_rule(data):
    signed_data = data.astype('float64')*2 - 1
    weights = np.cov(np.transpose(signed_data), bias=True)
    thresholds = -np.mean(signed_data, axis=0)
    return weights, thresholds
