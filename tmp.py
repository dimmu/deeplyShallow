import tensorflow as tf
from logReg import LogReg
import numpy as np
import os
import time
import datetime
from TextCNN import TextCNN
from TextCNN2 import TextCNN2
from word2vec import EmbeddingModel
from word2vec2 import EmbeddingModel2
from loadData import loadEmo, BuildDictionary, VocabVec, fastWordVec
from BiLSTM import BiLSTM
evaluate_every=100

# Training
# ==================================================
(trainText, trainY, valText, valY, testText, testY) = loadEmo()
vocab = BuildDictionary(trainText,20)

input_size = len(vocab)
num_classes = 2
l2_lambda = 0.0001
batch_size = 128
step_size = int(np.floor(trainText.size/batch_size))
N_LOOPS=25
def train_step(model,session,x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
    }
    _, loss, accuracy = sess.run(
    [train_op,model.loss, model.accuracy],feed_dict)
    time_str = datetime.datetime.now().isoformat()
    #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    print("{}:, loss {:g}, acc {:g}".format(time_str, loss, accuracy))
    #train_summary_writer.add_summary(summaries, step)

def dev_step(model, session, x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
    }
    _, loss, accuracy = sess.run(
    [global_step, model.loss, model.accuracy],
    feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("VAL: {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
    return accuracy
    #if writer:
    #    writer.add_summary(summaries, step)




with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        #model = LogReg(input_size=1000, num_classes=2, l2_lambda=l2_lambda)
        # model = TextCNN(sequence_length=20,num_classes=2,vocab_size=len(vocab)+1,
        #     embedding_size=100,
        #     filter_sizes=[3,5,9],
        #     num_filters=128,
        #     l2_reg_lambda=0.000)
        # model = TextCNN2(sequence_length=20,num_classes=2,vocab_size=len(vocab)+1,
        #     embedding_size=100,
        #     filter_size=3,
        #     num_filters=64,
        #     l2_reg_lambda=0.000)

        model= EmbeddingModel2(batch_size=128,sequence_length=20,num_classes=2,
             vocab_size=len(vocab)+1,embedding_size=150,l2_reg_lambda=0.0001)

        # model = BiLSTM(batch_size=batch_size, times_steps=19,sequence_length=20,num_classes=2,
        #     vocab_size=len(vocab)+1,embedding_size=50,hidden_size=100)
        
        # seqlens=[]
        # for tmp in range(0,batch_size):
        #     seqlens.append(20)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        for loop in range(0, N_LOOPS):
            trainIdx = np.random.permutation(len(trainY))
            for i in range(0,step_size):
                #x_batch, y_batch = zip(*batch)
                thisIdx=trainIdx[i:len(trainIdx):step_size]
                #x_batch = fastWordVec(trainText[thisIdx],1000)
                x_batch = VocabVec(trainText[thisIdx], vocab, doc_length=20)
                y_batch = trainY[thisIdx]
                train_step(model,sess,x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
            print("***********************")
            print("DONE WITH ITERATION ", loop)
            print("***********************")
            valIdx=list(range(0,len(valText)))
            repeatVals=int(np.floor(len(valY)/batch_size))
            allAc=[]
            for valStep in range(0,repeatVals):
                thisIdx=valIdx[valStep:len(valY):repeatVals]
                #x_batch = fastWordVec(valText[thisIdx],1000)
                x_batch = VocabVec( valText[thisIdx], vocab, doc_length=20)
                y_batch = valY[thisIdx]
                #dev_step(model, sess, x_batch, y_batch)
                allAc.append(dev_step(model, sess, x_batch, y_batch))
            tt = sum(allAc)/len(allAc)
            print("ACCURACY:", tt)
            print("**********************")