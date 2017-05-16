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
# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     trainX = tf.placeholder(tf.float32, shape=(batch_size, dictionary_size))
#     trainY = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     valX = tf.placeholder(tf.float32, shape=(None, dictionary_size))
#     valY = tf.placeholder(tf.float32, shape=(None, num_labels))
  
#     #produce output... 
#     weights = tf.Variable(tf.truncated_normal([dictionary_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))
#     logits = tf.matmul(trainX, weights) + biases
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=trainY, logits=logits) +
#         lam * tf.nn.l2_loss(weights))
#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)
#     valid_h= tf.nn.relu(tf.matmul( tf_valid_dataset, relu_weights) +relu_biases)
#     valid_1_h = tf.nn.relu(tf.matmul(valid_h,relu1_weights)+relu1_biases)
#     valid_prediction = tf.nn.softmax(
#     tf.matmul(valid_1_h, weights) + biases)
#     test_h= tf.nn.relu(tf.matmul( tf_test_dataset, relu_weights) +relu_biases)
#     test_1_h = tf.nn.relu(tf.matmul(test_h,relu1_weights)+relu1_biases)
#     test_prediction = tf.nn.softmax(tf.matmul(test_1_h, weights) + biases)