"""
Train a Hopfield network to classify MNIST digits.

Since Hopfield networks are not supervised models, we must
turn classification into a memory recall task. To do this,
we feed the network augmented vectors containing both the
image and a one-hot vector representing the class.
"""

import time

from hopfield import Network, extended_storkey_update
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100

MNIST = input_data.read_data_sets('MNIST_data', one_hot=True)

def main():
    """
    Train the model and measure the results.
    """
    network = Network(28*28 + 10)
    with tf.Session() as sess:
        print('Training...')
        train(sess, network, MNIST.train)
        print('Evaluating...')
        print('Validation accuracy: ' + str(accuracy(sess, network, MNIST.validation)))
        print('Testing accuracy: ' + str(accuracy(sess, network, MNIST.test)))
        print('Training accuracy: ' + str(accuracy(sess, network, MNIST.train)))

def train(sess, network, dataset):
    """
    Train the Hopfield network.
    """
    image_ph = tf.placeholder(tf.float32, shape=(28*28,))
    label_ph = tf.placeholder(tf.bool, shape=(10,))
    joined = tf.concat((tf.greater_equal(image_ph, 0.5), label_ph), axis=-1)
    update = extended_storkey_update(joined, network.weights)
    sess.run(tf.global_variables_initializer())
    i = 0
    start_time = time.time()
    for label, image in zip(dataset.labels, dataset.images):
        sess.run(update, feed_dict={image_ph: image, label_ph: label})
        i += 1
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            frac_done = i / len(dataset.images)
            remaining = elapsed * (1-frac_done)/frac_done
            print('Done %.1f%% (eta %.1f minutes)' % (100 * frac_done, remaining/60))

def accuracy(sess, network, dataset):
    """
    Compute the test-set accuracy of the Hopfield network.
    """
    images_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28*28))
    preds = classify(network, images_ph)
    num_right = 0
    for i in range(0, len(dataset.images), BATCH_SIZE):
        images = dataset.images[i : i+BATCH_SIZE]
        labels = dataset.labels[i : i+BATCH_SIZE]
        preds_out = sess.run(preds, feed_dict={images_ph: images})
        num_right += np.dot(preds_out.flatten(), labels.flatten())
    return num_right / len(dataset.images)

def classify(network, images):
    """
    Classify the images using the Hopfield network.

    Returns:
      A batch of one-hot vectors.
    """
    numeric_vec = tf.cast(tf.greater_equal(images, 0.5), tf.float32)*2 - 1
    thresholds = network.thresholds[-10:]
    logits = tf.matmul(numeric_vec, network.weights[:28*28, -10:]) - thresholds
    return tf.one_hot(tf.argmax(logits, axis=-1), 10)

if __name__ == '__main__':
    main()
