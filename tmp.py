import tensorflow as tf
from logReg import LogReg
import numpy as np
import os
import time
import datetime

evaluate_every=100

# Training
# ==================================================
input_size=10000
num_classes=2
l2_lambda=0.01
batch_size=128
step_size=int(np.floor(trainText.size/batch_size))

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        LR = LogReg(input_size=input_size, num_classes=2, l2_lambda=l2_lambda)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(LR.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                LR.input_x: x_batch,
                LR.input_y: y_batch,
            }
            _, loss, accuracy = sess.run(
                [train_op,LR.loss, LR.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print("{}:, loss {:g}, acc {:g}".format(time_str, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                LR.input_x: x_batch,
                LR.input_y: y_batch,
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, LR.loss, LR.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches Training loop. For each batch...
        for i in range(0,step_size):
            #x_batch, y_batch = zip(*batch)
            thisIdx=trainIdx[i:len(trainIdx):step_size]
            x_batch = fastWordVec(text[thisIdx],input_size)
            y_batch = Y[thisIdx]
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            # if current_step % evaluate_every == 0:
            #     print("\nEvaluation:")
            #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
            #     print("")
            # if current_step % FLAGS.checkpoint_every == 0:
            #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #     print("Saved model checkpoint to {}\n".format(path))