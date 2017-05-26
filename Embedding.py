import tensorflow as tf
import numpy as np
#word2vec model

class EmbeddingModel(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, batch_size, sequence_length, num_classes, vocab_size,
      embedding_size, l2_reg_lambda=0.0, dropout_keep_prob=0.5):

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
            print("INPUT_X", self.input_x.shape)
            print("embedded_chars", self.embedded_chars.shape)
            #self.agx=tf.reduce_sum(self.embedded_chars, [1])
            self.agx=tf.reshape(self.embedded_chars,[-1,sequence_length*embedding_size])
            print("AGX:",self.embedded_chars.shape)
            self.dd=tf.nn.dropout(self.agx,dropout_keep_prob)
            self.w_out=tf.Variable(tf.zeros([embedding_size*sequence_length,num_classes],name="output_w"))
            self.b = tf.ones([num_classes])

        with tf.name_scope("forward"):
            logits = tf.matmul(self.agx, self.w_out) + self.b
            print("logits ",logits.shape)
            self.scores = tf.nn.softmax(logits)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, 
                                                            labels=self.input_y)
            l2_loss += tf.nn.l2_loss(self.w_out)
            l2_loss += tf.nn.l2_loss(self.b)
            l2_loss += tf.nn.l2_loss(self.W)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            #loss_summary=tf.summary.scalar(self.loss)
            #acc_summary=tf.summary.scalar(self.accuracy)
            #self.summaries=tf.summary.merge(loss_summary,acc_summary)