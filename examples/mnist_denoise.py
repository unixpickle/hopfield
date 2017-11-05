"""
Train a Hopfield network to reconstruct MNIST digits, then
use it to gradually remove noise from images.
"""

from hopfield import Network, hebbian_update
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
MNIST = input_data.read_data_sets('MNIST_data', one_hot=True)

def main():
    """
    Train the model and draw the results.
    """
    network = Network(28 * 28)
    with tf.Session() as sess:
        print('Training...')
        train(sess, network, MNIST.train.images)
        print('Denoising...')
        noisy = noisy_images(MNIST.train.images[:10])
        plt.ion()
        for batch in iterate_network(sess, network, noisy):
            plt.imshow(batch.reshape((len(batch)*28, 28)))
            plt.pause(1)

def train(sess, network, dataset):
    """
    Train the Hopfield network.
    """
    images_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28*28))
    update = hebbian_update(tf.greater_equal(images_ph, 0.5), network.weights)
    sess.run(tf.global_variables_initializer())
    for i in range(0, len(dataset), BATCH_SIZE):
        images = dataset[i : i+BATCH_SIZE]
        sess.run(update, feed_dict={images_ph: images})

def noisy_images(images):
    """
    Binarize the images and add noise to them.
    """
    noise = np.random.normal(size=images.shape) > 2
    return (images > 0.5) ^ noise

def iterate_network(sess, network, images):
    """
    Continually apply the Hopfield network to the images.

    Returns:
      An iterator over batches of images. Starts with the
        passed batch of images.
    """
    images_ph = tf.placeholder(tf.bool, shape=images.shape)
    output = network.step(images_ph)
    while True:
        yield images
        images = sess.run(output, feed_dict={images_ph: images})

if __name__ == '__main__':
    main()
