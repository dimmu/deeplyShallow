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

#evaluate_every=100


parser = OptionParser()
parser.add_option("-m", "--model", dest="modelType",
                  help="model type to use bow, emb, emb2, cnn,cnn2,", metavar="FILE", 
                  default="emb")
parser.add_option("-v", "--verbose",
                  dest="evaluate_every", default=100,
                  help="how often to print evaluations to stdout")

(options, args) = parser.parse_args()

#print("options", options)
evaluate_every=int(options.evaluate_every)
modelType=options.modelType
# Data loading and init processing
# ==================================================

(trainText, trainY, valText, valY, testText, testY) = loadEmo()
minFreq=20
doc_length = 20
num_classes = 2
l2_lambda = 0.0001
batch_size = 128

vocab = BuildDictionary(trainText,20)
input_size = len(vocab)
step_size = int(np.floor(trainText.size/batch_size))
N_LOOPS=25


timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs",modelType, timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")


def train_step(model,session,x_batch, y_batch, stepNum, print_out):
    """
    A single training step
    """
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
    }
    _, loss, accuracy = sess.run(
    [train_op,model.loss, model.accuracy],feed_dict)
    if print_out == True:
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, stepNum, loss, accuracy))

def val_step(model, session, x_batch, y_batch, writer=None):
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
    return (accuracy,loss)





with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():

        #initialize summary writers
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        if modelType == "logreg":
            model = LogReg(input_size=1000, num_classes=2, l2_lambda=l2_lambda)
        elif modelType == "cnn":
            model = TextCNN(sequence_length=20,num_classes=2,vocab_size=len(vocab)+1,
                 embedding_size=100,
                 filter_sizes=[3,5,9],
                 num_filters=128,
                l2_reg_lambda=0.000)
        elif modelType == "cnn2": 
            model = TextCNN2(sequence_length=20,num_classes=2,vocab_size=len(vocab)+1,
                 embedding_size=100,
                 filter_size=3,
                 num_filters=64,
                 l2_reg_lambda=0.000)
        elif modelType == "emb":
            model= EmbeddingModel(batch_size=128,sequence_length=20,num_classes=2,
                vocab_size=len(vocab)+1,embedding_size=50,l2_reg_lambda=0.00)
        elif modelType == "emb2": 
            model= EmbeddingModel2(batch_size=128,sequence_length=20,num_classes=2,
              vocab_size=len(vocab)+1,embedding_size=50,l2_reg_lambda=0.0001)

        # model = BiLSTM(batch_size=batch_size, times_steps=19,sequence_length=20,num_classes=2,
        #     vocab_size=len(vocab)+1,embedding_size=50,hidden_size=100)

        # Define Training procedure
        
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        # Initialize all variables
        stepNum=0
        sess.run(tf.global_variables_initializer())
        for loop in range(0, N_LOOPS):
            trainIdx = np.random.permutation(len(trainY))
            for i in range(0,step_size):
                stepNum+=1
                thisIdx=trainIdx[i:len(trainIdx):step_size]
                #x_batch = fastWordVec(trainText[thisIdx],1000)
                x_batch = VocabVec(trainText[thisIdx], vocab, doc_length=20)
                y_batch = trainY[thisIdx]
                if (i%evaluate_every) ==0:
                    train_step(model,sess,x_batch, y_batch,stepNum,True)
                else:
                    train_step(model,sess,x_batch,y_batch,stepNum,False)
                current_step = tf.train.global_step(sess, global_step)

            valIdx=list(range(0,len(valText)))
            repeatVals=int(np.floor(len(valY)/batch_size))
            repeatTests=repeatVals
            allAc=[]
            allAcTest=[]
            allLoss=[]
            allLossTest=[]
            for valStep in range(0,repeatVals):
                thisIdx=valIdx[valStep:len(valY):repeatVals]
                x_batch = VocabVec( valText[thisIdx], vocab, doc_length=20)
                y_batch = valY[thisIdx]
                (valAcc,valLoss)=val_step(model,sess,x_batch,y_batch)
                allAc.append(valAcc)
                allLoss.append(valLoss)
                x_batch_test= VocabVec( testText[thisIdx], vocab, doc_length=20)
                y_batch_test= testY[thisIdx]
                (valAccTest,valLossTest)=val_step(model,sess,x_batch_test,y_batch_test)
                allAcTest.append(valAccTest)
                allLossTest.append(valLossTest)
            meanAcc=np.mean(allAc)
            meanLoss=np.mean(allLoss)
            meanAccTest=np.mean(allAcTest)
            meanLossTest=np.mean(allLossTest)
            print("VALIDATION: {}: loss {:g}, acc {:g}".format(loop, meanLoss, meanAcc))
            print("TEST: {}: loss {:g}, acc {:g}".format(loop, meanLossTest, meanAccTest))
            loss_summary = tf.summary.scalar("loss", meanLoss)
            acc_summary = tf.summary.scalar("accuracy", meanAcc)
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            