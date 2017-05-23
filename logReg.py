import tensorflow as tf

# batch_size=128
# dictionary_size=10000

class LogReg(object):
    def __init__(self, input_size, num_classes, l2_lambda):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, input_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_lambda = tf.Variable(l2_lambda)
        #self.w = tf.Variable(tf.zeros([input_size, num_classes]), name="W")
        self.w = tf.Variable(tf.random_uniform([input_size, num_classes], -1.0, 1.0), name="W")
        self.b = tf.Variable(tf.zeros([num_classes]))
        l2_loss = tf.Variable(0.0)
        with tf.name_scope("forward"):
            logits = tf.matmul(self.input_x, self.w) + self.b
            self.scores = tf.nn.softmax(logits)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, 
                                                            labels=self.input_y)
            l2_loss += tf.nn.l2_loss(self.w)
            l2_loss += tf.nn.l2_loss(self.b)
            self.loss = tf.reduce_mean(losses) + l2_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")