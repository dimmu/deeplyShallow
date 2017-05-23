import tensorflow as tf
import numpy as np


class TextCNN2(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_size, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = tf.constant(0.5)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print("SIZE OF EMBEDDED CHARS", self.embedded_chars.shape)
            print("embeded chars expanded:",self.embedded_chars_expanded.shape)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
            conv1 = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            print("CONV1 shape", conv1.shape)
            # Apply nonlinearity
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
            print("h_1 size:", h1.shape)

            filter2_shape=[filter_size, num_filters,1, num_filters]
            W2 = tf.Variable(tf.truncated_normal(filter2_shape, stddev=0.1), name="W2")
            b2=tf.Variable(tf.constant(0.1, shape=[num_filters],name="b2"))
            conv2 = tf.nn.conv2d(
                tf.transpose(conv1,[0,1,3,2]),
                W2,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv2")
            print("CONV2...", conv2.shape)
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
            pooled2 = tf.nn.max_pool(
                h2,
                ksize=[1, sequence_length -2* filter_size + 2, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool2")
            pooled_outputs.append(pooled2)
            print("POOLED 2 size", pooled2)
        # Combine all the pooled features
        num_filters_total = num_filters #* len(filter_size)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")