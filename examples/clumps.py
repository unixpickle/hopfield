"""
Train a Hopfield network on three "clumps":

      <1, 1, 0, 0, 0, 0>
      <0, 0, 1, 1, 0, 0>
      <0, 0, 0, 0, 1, 1>

Used as a sanity check.
"""

from hopfield import Network, hebbian_update
import numpy as np
import tensorflow as tf

def main():
    """
    Make sure a Hopfield network can learn non-overlapping
    clumps of bits.
    """
    network = Network(6)
    samples = tf.constant(np.array([[True, True, False, False, False, False],
                                    [False, False, True, True, False, False],
                                    [False, False, False, False, True, True]]))
    update = hebbian_update(samples, network.weights)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(update)
        input_ph = tf.placeholder(tf.bool, shape=(6,))
        converged = network.step(network.step(input_ph))
        print(sess.run(converged, feed_dict={input_ph: [True]+[False]*5}))
        print(sess.run(converged, feed_dict={input_ph: [False]*5+[True]}))

if __name__ == '__main__':
    main()
