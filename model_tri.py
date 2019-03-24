import re
import codecs
import numpy as np
import tensorflow as tf


def create_model_tri(sess, config, id_to_char):
    model = Model_tri(config)
    sess.run(tf.global_variables_initializer())
    emb_weights = sess.run(model.char_lookup.read_value())
    emb_weights = load_word2vec("data/100.utf8", id_to_char, 100, emb_weights)
    sess.run(model.char_lookup.assign(emb_weights))
    return model


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    new_weights = old_weights
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[re.sub('\d', '0', word.lower())]
            c_zeros += 1
    return new_weights


class Model_tri(object):
    def __init__(self, config):
        self.num_char = config["num_char"]
        self.sen_len = config["sen_len"]
        self.char_dim = config["char_dim"]
        self.batch = config["tri_batch"]
        self.trigger_num = config["tri_num"]
        self.pf_dim = config["tri_pf_dim"]
        self.window = config["tri_window"]
        self.feature = config["tri_feature"]
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.global_step = tf.Variable(0, trainable=False)

        self.keep_prob = tf.placeholder(tf.float32)
        self.cuts = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.masks = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        # self.trigger_sen = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.trigger_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.tri_loc_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.char_lookup = tf.get_variable(shape=[self.num_char, self.char_dim], initializer=self.initializer, name="tri_char_lookup")

        emb, lxl = self.embedding_layer(self.char_inputs, self.tri_loc_inputs, self.char_lookup, self.cuts)
        conved = self.convolution_layer(emb)
        pooled = self.dynamic_layer(conved, self.masks)
        output = self.output_layer(lxl, pooled)
        self.max = tf.arg_max(output, dimension=1)
        self.loss = self.loss_layer(output, self.trigger_inputs)
        self.train_step = tf.train.AdadeltaOptimizer(rho=0.95, epsilon=0.000001).minimize(self.loss)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def run_step(self, sess, batch):
        chars, ls, tri, arg, tri_in, tri_loc, mask, cut = batch
        feed_dict = {
            self.keep_prob: 0.5,
            self.cuts: np.asarray(cut),
            self.masks: np.asarray(mask),
            self.char_inputs: np.asarray(chars),
            self.trigger_inputs: np.asarray(tri_in),
            self.tri_loc_inputs: np.asarray(tri_loc),
        }
        global_step, loss, max, _ = sess.run([self.global_step, self.loss, self.max, self.train_step], feed_dict)
        return global_step, loss, max

    def embedding_layer(self, char_inputs, tri_loc_inputs, char_lookup, cuts):
        with tf.variable_scope("tri_embedding", reuse=tf.AUTO_REUSE):
            embed, lxl = list(), list()                                  # char_lookup: [20136, char_dim] char_inputs: [batch_size, sen_len]
            cwf = tf.nn.embedding_lookup(char_lookup, char_inputs)       # [batch, sen_len, char_dim]
            pf_lookup = tf.get_variable(name="tri_pf_lookup", shape=[2*self.sen_len-1, self.pf_dim], dtype=tf.float32, initializer=self.initializer)
            pf = tf.nn.embedding_lookup(pf_lookup, tri_loc_inputs)       # [batch, sen_len, pf_dim]
            embed = tf.concat([cwf, pf], -1)                             # [batch, sen_len, char_dim + pf_dim]
            for i in range(self.batch):
                lxl_0 = tf.slice(cwf, begin=[i, cuts[i]-1, 0], size=[1, 3, self.char_dim])
                lxl.append(lxl_0)
            lxl = tf.squeeze(lxl)                                        # [batch, 3, char_dim]
            lxl = tf.reshape(lxl, shape=[self.batch, 3*self.char_dim])   # [batch, 3*char_dim]
            return embed, lxl

    def convolution_layer(self, emb):
        with tf.variable_scope("tri_convolution", reuse=tf.AUTO_REUSE): # emb: [batch, sen_len, char_dim + pf_dim]
            w = tf.get_variable(name="tri_conv_w", shape=[self.window, self.char_dim+self.pf_dim, self.feature], dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv1d(emb, w, stride=1, padding="VALID")      # [batch, sen_len-window+1, feature]
            b = tf.get_variable(name="tri_conv_b", shape=[self.sen_len-self.window+1, self.feature], dtype=tf.float32, initializer=self.initializer)
            conved = tf.add(tf.nn.tanh(conv), b)                        # [batch, sen_len-window+1, feature]
            return conved

    def dynamic_layer(self, conv, mask):
        with tf.variable_scope("tri_pool", reuse=tf.AUTO_REUSE):
            mask_embedding = tf.constant([[0, 0], [1, 0], [0, 1]], dtype=tf.float32)  # [3, 2]
            mask = tf.nn.embedding_lookup(mask_embedding, mask)                       # [batch, sen_len-window+1, 2]
            pooled = tf.reduce_max(tf.expand_dims(mask*100, 2) + tf.expand_dims(conv, 3), axis=1) - 100
            return tf.reshape(pooled, [-1, 2*self.feature])                           # [batch, 2*feature]

    def output_layer(self, lxl, pooled):
        with tf.variable_scope("tri_output"):
            f = tf.concat([lxl, pooled], axis=-1)                   # [batch, 3*char_dim + 2*feature_map]
            w_s = tf.get_variable(name="tri_out_w", shape=[3*self.char_dim + 2*self.feature, self.trigger_num], dtype=tf.float32, initializer=self.initializer)
            b_s = tf.get_variable(name="tri_out_b", shape=[self.trigger_num], dtype=tf.float32, initializer=self.initializer)
            output = tf.nn.bias_add(tf.matmul(f, w_s), b_s)         # [batch, trigger_num]
            return output

    @staticmethod
    def loss_layer(tri_pre, trigger_inputs):
        with tf.variable_scope("tri_loss", reuse=tf.AUTO_REUSE):
            trigger_inputs = tf.cast(trigger_inputs, tf.float32)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tri_pre, labels=trigger_inputs, name="loss"))
            return loss
