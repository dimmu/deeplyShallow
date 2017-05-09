import tensorflow as tf
import pandas as pd
import vocab
import numpy as np
import time
import os
from copy import deepcopy

# ptb_word_lm.py and cs224d problem sets were used
# as inspiration for code layout

class Config(object):
    def __init__(self, vocab, max_seq_len):
        self.hidden_size = 50
        self.epoch_fraction = .1
        self.max_epoch = 100
        self.dropout = 0.9
        self.lr = .0001
        self.layers = 2
        self.label_size = 2
        self.should_cell_dropout = True
        self.should_input_dropbox = True
        self.embed_size = 50
        self.vocab = vocab
        self.batch_size = 1
        self.max_seq_len = max_seq_len
        self.anneal_threshold = 0.99
        self.anneal_by = 1.5
        self.l2 = 0.02


class LSTMSentiment(object):
    def __init__(self, is_training, config):
        self.input_lengths = tf.placeholder(tf.int32,[config.batch_size])
        self.input_data = tf.placeholder(dtype=tf.int32,
                                    shape=(config.batch_size,
                                           config.max_seq_len))
        self.label = tf.placeholder(dtype=tf.int32)
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=(config.max_seq_len))
        self.lr = tf.placeholder(dtype=tf.float32)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_size,
                                            forget_bias=0.0)
        if is_training and config.should_cell_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                                output_keep_prob=config.dropout)
        if config.layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*config.layers)

        initial_state = cell.zero_state(batch_size=config.batch_size,
                                        dtype=tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(name="embedding",
                                        shape=(config.vocab.size,
                                               config.embed_size))
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training and config.should_input_dropbox:
            inputs = tf.nn.dropout(inputs, keep_prob=config.dropout)

        state = initial_state
        # this causes a runtime error
        # break apart the input
        #inputs = [tf.squeeze(input=t, squeeze_dims=[1]) for t in
        #          tf.split(value=_inputs, split_dim=[1],
        #                   num_split=config.max_seq_len)]
        #outputs, states = tf.nn.rnn(cell=cell,
        #          inputs=inputs,
        #          initial_state=initial_state,
        #          sequence_length=self.input_lengths)

        # unrolled call to tf.nn.rnn sort of
        outputs = []
        with tf.variable_scope("rnn") as scope:
            for cycle in range(config.max_seq_len):
                if cycle > 0:
                    scope.reuse_variables()
                output, state = cell(inputs[:,cycle,:], state)
                outputs.append(output)

        with tf.variable_scope("softmax"):
            W = tf.get_variable(name="W",
                                shape=(config.hidden_size, config.label_size))
            b = tf.get_variable(name="b",
                                shape=(config.label_size))
            self.logits = tf.matmul(tf.squeeze(tf.pack(outputs),[1]), W) + b

        self.pred = tf.argmax(self.logits, 1)

        loss_fnc = tf.nn.sparse_softmax_cross_entropy_with_logits
        self.losses = loss_fnc(self.logits, self.labels)
                     #config.l2 * tf.nn.l2_loss(W))
        self.loss = tf.reduce_mean(self.losses)

        if not is_training:
            return

        self.lr = tf.Variable(config.lr, trainable=False)
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

    def set_lr(self, sess, new_lr):
        sess.run(tf.assign(self.lr, new_lr))


def run_epoch(config, sess, model, data, opt, suffix=None):
    start_time = time.time()
    loss = 0.0
    epoch_size = int(len(data) * config.epoch_fraction)
    indexes = np.random.randint(0, len(data), epoch_size)
    correct = 0.0
    for index in indexes:
        # seq should be (1 x n)
        # batch size of 1 (at the moment)
        entry = data[index]
        seq = [0] * config.max_seq_len
        seq_len = len(entry[1])
        # drop input that is too long
        if seq_len > config.max_seq_len:
            seq_len = config.max_seq_len
            seq = entry[1][:config.max_seq_len]
        else:
            seq[:seq_len] = entry[1]
        seqs = [seq]
        labels = [entry[0]]
        labels_big = [-1] * config.max_seq_len
        labels_big[:seq_len] = [entry[0]] * seq_len

        l, pred, _ = sess.run([model.loss, model.pred, opt],
                              {model.input_lengths: [seq_len],
                               model.input_data: seqs,
                               model.label: labels,
                               model.labels: labels_big})
        loss += l
        if pred[seq_len - 1] == entry[0]:
            correct += 1

    runtime = time.time() - start_time
    return loss, runtime, correct/epoch_size


def clean_string(s):
    s = s.lower()
    s = s.replace(".", "")
    s = s.replace("\\", "")
    s = s.replace("\"", "")
    s = s.replace("'", "")
    s = s.replace(",", "")
    s = s.replace(":", "")
    s = s.replace(";", "")
    s = s.replace("-", "")
    return s


def clean_data(d):
    for index in range(len(d)):
        d[index, 1] = clean_string(d[index,1])
    return d


def create_vocab(sentence_list):
    big = ""
    max_seq_len = 0
    for s in sentence_list:
        seq_len = len(s.split())
        if seq_len > max_seq_len:
            max_seq_len = seq_len
        big += s + " "
    return vocab.Vocab(big), max_seq_len

def generate_data(config, data_with_strings):
    a = []
    for i in range(len(data_with_strings)):
        label = data_with_strings[i, 0]
        words = data_with_strings[i, 1].split()
        a.append([label, [config.vocab.getindex(w) for w in words]])

    return a

def run():
    dataset = "airline"
    train = pd.read_csv('data/' + dataset + "_train.tsv",
                      delimiter='~').as_matrix()
    train = clean_data(train)
    validate = pd.read_csv('data/' + dataset + "_validate.tsv",
                       delimiter='~').as_matrix()
    validate = clean_data(validate)
    #test = pd.read_csv('data/' + dataset + "_test.tsv",
    #                   delimiter='~').as_matrix()

    vocab, seq_len = create_vocab(train[:,1])
    print("max sequence length is %d" % seq_len)
    config = Config(vocab, seq_len)
    config_notrain = deepcopy(config)
    train_data = generate_data(config, train)
    validate_data = generate_data(config, validate)

    with tf.Session() as sess:
        with tf.variable_scope("model"):
            s = LSTMSentiment(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True):
            s_notrain = LSTMSentiment(is_training=False, config=config_notrain)

        tf.initialize_all_variables().run()
        best_val_loss = float('inf')
        prev_loss = float('inf')
        for i in range(config.max_epoch):
            # all this printing is because stream redirects
            # didn't work in screen on my ubuntu gcp node
            out_str = "starting epoch %d\n" % i
            loss, runtime, accuracy = run_epoch(config,
                                                sess,
                                                s,
                                                train_data,
                                                s.opt,
                                                i)
            out_str += ("loss is: %f\n"
                        "epoch runtime: %f\n"
                        "accuracy: %f\n") % (
                            loss, runtime, accuracy)

            if loss > prev_loss * config.anneal_threshold:
                config.lr /= config.anneal_by
                s.set_lr(sess, config.lr)
                out_str += 'annealed lr to %f' % config.lr
            prev_loss = loss

            val_loss, val_runtime, val_accuracy = run_epoch(config,
                                                            sess,
                                                            s_notrain,
                                                            validate_data,
                                                            tf.no_op())
            out_str += ("val loss is: %f\n"
                        "val epoch runtime: %f\n"
                        "val accuracy: %f\n") % (
                            val_loss, val_runtime, val_accuracy)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                out_str += "saving better epoch %d\n" % i
                saver.save(sess, './weights/epoch.%s.temp' % i)

	    # print the output and save to file since pipe redirect doesn;t work...
	    print(out_str)
	    if not os.path.isfile("outputs.txt"):
		with open("outputs.txt", "w") as f:
		    f.write(out_str)
	    else:
		with open("outputs.txt", "a") as f:
		    f.write(out_str)

if __name__ == "__main__":
    run()
