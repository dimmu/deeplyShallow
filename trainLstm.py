import tensorflow as tf
from logReg import LogReg
import numpy as np
import os
import time
import datetime
from TextCNN import TextCNN
from TextCNN2 import TextCNN2
from Embedding import EmbeddingModel
from Embedding2 import EmbeddingModel2
from loadData import loadEmo, BuildDictionary, VocabVec, fastWordVec
from BiLSTM import BiLSTM
from optparse import OptionParser
from tensorflow.contrib import rnn
import sonnet as snt

def transform(text, vocab, max_len=150):
    #rtn = np.zeros((max_len,1))
    rtn = []
    for i,c in enumerate(text):
        if i<max_len:
            rtn.append(ord(c))
            #rtn[i]=ord[c]
    if i<max_len:
        for i in range(i+1,max_len):
            rtn.append(0)
    return rtn

def transform_one_hot(text, vocab, max_len=150):
    rtn = np.zeros((max_len,len(vocab) +1))
    for i,c in enumerate(text):
        if i<max_len:
            rtn[i,vocab[c]]=1
    return rtn


class BatchLoader:
    
    def __init__(self, X, Y, batch_size, vocab, max_len):
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        self.batch_size = batch_size
        self.max_batch = int(np.floor(len(Y) / self.batch_size))
        self.epoch = 0
        self.batch = 0
        self.vocab = vocab
        self.max_len = max_len
        self.reshuffle()

    def get_batch(self, epoch):
        if epoch != self.epoch:
            self.reshuffle()
            self.epoch += 1
            self.batch  = 0

        start = self.batch*self.batch_size
        stop = min( (self.batch+1)*self.batch_size , len(self.Y))

        # Never get out
        self.batch = (self.batch+1) % self.max_batch
        rtn = []
        for i in range(start,stop):
            #rtn.append( transform(self.X[i],self.vocab,self.max_len))
            rtn.append( transform(self.X[i],self.vocab))
        return (np.array(rtn), self.Y[start:stop])

    def reshuffle(self):
        permIdx = np.random.permutation(len(self.Y))
        self.X=self.X[permIdx]
        self.Y=self.Y[permIdx]



(trainText, trainY, valText, valY, testText, testY) = loadEmo()


#padding stuff
max_len = 0
vocab = {}
for text in trainText:
    if len(text)>max_len:
        max_len=len(text)
    for c in text:
        if c not in vocab:
            vocab[c] = len(vocab)+1

batch_size=128
bl = BatchLoader(trainText,trainY,batch_size,vocab,max_len)
num_classes = 2
embedding_size=20
num_hidden=128
learning_rate=0.003
max_len=150

num_relu_weights=128

#input_x = tf.placeholder(tf.float32, [None, max_len,len(vocab)+1], name="input_x")
input_x = tf.placeholder(tf.int32, [None, max_len], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
print('input_x', input_x.shape)

embedding_matrix = tf.Variable(tf.random_uniform([512, embedding_size], -1.0, 1.0),name="W")
relu_weights = tf.Variable(0.01*tf.random_normal([2*num_hidden, num_relu_weights]))
weights_out = tf.Variable(0.01*tf.random_normal([num_relu_weights, num_classes]))
bias_out = tf.Variable(tf.random_normal([num_classes]))
bias_relu = tf.Variable(tf.random_normal([num_relu_weights]))
def embed(X, embedding_matrix):
    return tf.nn.embedding_lookup(embedding_matrix, X)

def bilstm(X, namespace='layer_0', return_sequences=False, dropout_pc=0.0, 
            bidirectional = True, n_layers=1):
        timesteps = max_len
        with tf.variable_scope(namespace):
            print('INPUT SHAPE', X.shape)
            X = tf.unstack(X, timesteps, 1)

            #lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            #lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            #outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X,
            #                                  dtype=tf.float32)
            #basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=2*num_hidden)
            #basic_cell = tf.contrib.rnn.GRUCell(num_units=2*num_hidden)
            #n_layers=2
            # layers = [tf.contrib.rnn.BasicLSTMCell(num_units=2*num_hidden)
            #             for layer in range(n_layers)]
            layers_fw = [tf.contrib.rnn.GRUCell(num_units=num_hidden)
                        for layer in range(n_layers)]

            layers_bw = [tf.contrib.rnn.GRUCell(num_units=num_hidden)
                        for layer in range(n_layers)]
            if dropout_pc>0.001 and dropout_pc <0.999:
                layers_fw = [tf.contrib.rnn.DropoutWrapper(layer, input_keep_prob=dropout_pc)
                            for layer in layers_fw]
                layers_bw = [tf.contrib.rnn.DropoutWrapper(layer, input_keep_prob=dropout_pc)
                            for layer in layers_bw]
            else:
                print("NO DROPOUT")
            multi_layer_cell_fw = tf.contrib.rnn.MultiRNNCell(layers_fw)
            multi_layer_cell_bw = tf.contrib.rnn.MultiRNNCell(layers_bw)
            #outputs, states = tf.nn.static_rnn(multi_layer_cell, X, dtype=tf.float32)
            outputs, states, _ = rnn.static_bidirectional_rnn(multi_layer_cell_fw,
                                                        multi_layer_cell_bw, X,
                                                        dtype=tf.float32)
            #print('OUTPUTS:',outputs.shape)
            #print('states:',states.shape)
            # layers =
            # outputs, states = tf.contrib.rnn.static_rnn(basic_cell, X,
            #                                     dtype=tf.float32)
        if return_sequences:
            return outputs
        else:
            return outputs[-1]


embeddings = embed(input_x, embedding_matrix)
# #lstm_states = bilstm(embeddings, return_sequences=True)
# #lstm_states = tf.transpose(lstm_states, [1, 0, 2])
# #lstm_state = bilstm(lstm_states,namespace='layer_1')
# lstm_state = build_lstm(embeddings,128, return_sequences=False)
#lstm_state = build_lstm(input_x,128, return_sequences=False)
lstm_state = bilstm(embeddings,namespace='layer_1')
lstm_drop = tf.nn.dropout(lstm_state, 0.5)
relu_out = tf.nn.relu(tf.matmul(lstm_drop,relu_weights)+bias_relu)
h_drop = tf.nn.dropout(relu_out,0.5)

prediction = tf.nn.softmax(tf.matmul(h_drop,weights_out)+bias_out)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=input_y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables and savers
init = tf.global_variables_initializer()

epochs = 100
training_steps = int(np.ceil(len(trainY)/batch_size))
best_validation_accuracy = 0
display_step=10

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    # val_dict = {
    #             X1: X1_val,
    #             X2: X2_val,
    #             Y: Y_val,
    #         }
    # test_dict = {
    #     X1: X1_test,
    #     X2: X2_test,
    #     Y: Y_test,
    # }
    for epoch in range(epochs):
        epoch_average_acc=0.0
        epoch_average_loss=0.0
        av_counter = 0
        bl.reshuffle()
        for step in range(training_steps):
          
        # Run optimization op (backprop)
            (x_batch, y_batch) = bl.get_batch(epoch)
            sess.run(train_op, feed_dict={input_x: np.squeeze(x_batch), input_y: y_batch})
            if step % display_step == 0 or step == 0:
            # Calculate batch loss and accuracy
            
                loss, acc = sess.run([loss_op, accuracy], feed_dict={input_x: np.squeeze(x_batch), input_y: y_batch})
                if av_counter == 0:
                    epoch_average_acc = acc
                    epoch_average_loss = loss
                    av_counter+=1
                else:
                    epoch_average_acc = (av_counter*epoch_average_acc + acc)/(av_counter+1)
                    epoch_average_loss = (av_counter*epoch_average_loss + loss)/(av_counter+1)
                    av_counter+=1
                print("Step " + str(step) + ", Avg Loss= " + \
                  "{:.4f}".format(epoch_average_loss) + ", Avg Accuracy= " + \
                  "{:.3f}".format(epoch_average_acc))
                

        print("Optimization Finished!")
        learning_rate = 0.01 * learning_rate
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #print("Validation Accuracy:", sess.run(accuracy, feed_dict=val_dict))
