"""
Implementation of the definition of Hopfield networks.
"""

import tensorflow as tf

class Network:
    """
    A Hopfield network.
    """
    def __init__(self, num_units, scope='hopfield'):
        # pylint: disable=E1129
        with tf.variable_scope(scope):
            self._weights = tf.get_variable('weights',
                                            shape=(num_units, num_units),
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
            self._thresholds = tf.get_variable('thresholds',
                                               shape=(num_units,),
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())

    @property
    def weights(self):
        """
        Get the weight 2-D Tensor for the network.

        Rows correspond to inputs and columns correspond
        to outputs.
        """
        return self._weights

    @property
    def thresholds(self):
        """
        Get the threshold 1-D Tensor for the network.
        """
        return self._thresholds

    def step(self, states):
        """
        Apply the activation rules to the states.

        Args:
          states: a 1-D or 2-D Tensor of input states.
            2-D Tensors represent batches of states.
            States must have dtype tf.bool.

        Returns:
          The new state Tensor after one timestep.
        """
        assert states.dtype == tf.bool

        numerics = 2*tf.cast(states, tf.float32) - 1
        if len(numerics.get_shape()) == 1:
            numerics = tf.expand_dims(numerics, axis=0)

        weighted_states = tf.matmul(numerics, self.weights)
        result = tf.greater_equal(weighted_states, self.thresholds)
        if len(states.get_shape()) == 1:
            return result[0]
        return result
