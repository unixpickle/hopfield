"""
Train a Hopfield network to classify MNIST digits.

Since Hopfield networks are not supervised models, we must
turn classification into a memory recall task. To do this,
we feed the network augmented vectors containing both the
image and a one-hot vector representing the class.
"""

from hopfield import Network, covariance_update
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
FORWARD_ITERS = 10
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
        print('Testing accuracy: ' + str(accuracy(sess, network, MNIST.validation)))
        print('Training accuracy: ' + str(accuracy(sess, network, MNIST.train)))

def train(sess, network, dataset):
    """
    Train the Hopfield network.
    """
    images_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28*28))
    labels_ph = tf.placeholder(tf.bool, shape=(BATCH_SIZE, 10))
    joined = tf.concat((tf.greater_equal(images_ph, 0.5), labels_ph), axis=-1)
    update = covariance_update(joined, network.weights, network.thresholds)
    sess.run(tf.global_variables_initializer())
    for i in range(0, len(dataset.images), BATCH_SIZE):
        images = dataset.images[i : i+BATCH_SIZE]
        labels = dataset.labels[i : i+BATCH_SIZE]
        sess.run(update, feed_dict={images_ph: images, labels_ph: labels})

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
