import tensorflow as tf
from logReg import LogReg
import numpy as np
import os
import time
import datetime
from loadData import loadEmo, fastWordVec
evaluate_every=100

# Training
# ==================================================
(trainText, trainY, valText, valY, testText, testY) = loadEmo()

input_size = 20000
num_classes = 2
l2_lambda = 0.01
batch_size = 128
step_size = int(np.floor(trainText.size/batch_size))
N_LOOPS=10
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




with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model = LogReg(input_size=input_size, num_classes=2, l2_lambda=l2_lambda)
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
                x_batch = fastWordVec(trainText[thisIdx],input_size)
                y_batch = trainY[thisIdx]
                train_step(model,sess,x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
        # def train_step(x_batch, y_batch):
        #     """
        #     A single training step
        #     """
        #     feed_dict = {
        #         model.input_x: x_batch,
        #         model.input_y: y_batch,
        #     }
        #     _, loss, accuracy = sess.run(
        #         [train_op,model.loss, model.accuracy],
        #         feed_dict)
        #     time_str = datetime.datetime.now().isoformat()
        #     #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        #     print("{}:, loss {:g}, acc {:g}".format(time_str, loss, accuracy))
        #     #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, LR.loss, LR.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches Training loop. For each batch...
        # trainIdx = np.random.permutation(len(trainY))
        # for i in range(0,step_size):
        #     #x_batch, y_batch = zip(*batch)
        #     thisIdx=trainIdx[i:len(trainIdx):step_size]
        #     x_batch = fastWordVec(trainText[thisIdx],input_size)
        #     y_batch = trainY[thisIdx]
        #     train_step(model,session,x_batch, y_batch)
        #     current_step = tf.train.global_step(sess, global_step)
            # if current_step % evaluate_every == 0:
            #     print("\nEvaluation:")
            #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
            #     print("")
            # if current_step % FLAGS.checkpoint_every == 0:
            #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #     print("Saved model checkpoint to {}\n".format(path))