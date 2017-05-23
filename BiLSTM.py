import numpy as np
import tensorflow as tf

class BiLSTM(object):
    def __init__(
        self,
        batch_size,
        times_steps,
        sequence_length,
        num_classes,
        vocab_size,
        embedding_size,
        hidden_size):
    
        self.input_x = tf.placeholder(tf.int32, shape=[batch_size,times_steps])
        self.input_y = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
        #seqlens for dynamic calculation
        #self.seqlen = tf.placeholder(tf.int32, shape=[batch_size]) 
    
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size],
                          -1.0, 1.0))
            embed = tf.nn.embedding_lookup(self.embeddings, self.input_x)
            print("EMBED:",embed.shape)
    
        with tf.name_scope("lstm"):
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,forget_bias=1.0)
            outputs, states = tf.nn.dynamic_rnn(self.lstm_cell, embed,
                                        sequence_length = sequence_length-2,
                                        dtype=tf.float32)

        self.weights = {
            'linear_layer': tf.Variable(tf.truncated_normal([hidden_size,
                num_classes], mean=0,stddev=.01))
                }
        self.biases = {
            'linear_layer':tf.Variable(tf.truncated_normal([num_classes],
                            mean=0,stddev=.01))
            }

        with tf.name_scope("forward"):
            logits = tf.matmul(states[1], self.weights["linear_layer"]) + self.biases["linear_layer"]
            self.scores = tf.nn.softmax(logits)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")  
        
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, 
                                                            labels=self.input_y)
            #l2_loss += tf.nn.l2_loss(self.w)
            #l2_loss += tf.nn.l2_loss(self.b)
            self.loss = tf.reduce_mean(losses) #+ l2_lambda * l2_loss
        
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")