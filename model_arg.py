import numpy as np
import tensorflow as tf
from model_tri import load_word2vec


def create_model_arg(sess, config, id_to_char):
    model = Model_arg(config)
    print("model!")
    sess.run(tf.global_variables_initializer())
    emb_weights = sess.run(model.char_lookup.read_value())
    emb_weights = load_word2vec("data/100.utf8", id_to_char, 100, emb_weights)
    sess.run(model.char_lookup.assign(emb_weights))
    print("model finished!")
    return model

class Model_arg(object):
    def __init__(self, config):
        self.num_char = config["num_char"]
        self.trigger_num = config["tri_num"]
        self.argrole_num = config["arg_num"]
        self.sen_len = config["sen_len"]
        self.char_dim = config["char_dim"]
        self.batch = config["arg_batch"]
        self.pf_dim = config["arg_pf_dim"]
        self.ef_dim = config["arg_ef_dim"]
        self.feature = config["arg_feature"]
        self.window = config["arg_window"]
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.global_step = tf.Variable(0, trainable=False)

        self.keep_prob = tf.placeholder(tf.float32)
        self.masks = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.trigger_sen = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.argrole_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.tri_loc_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.arg_loc_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.char_lookup = tf.get_variable(shape=[self.num_char, self.char_dim], initializer=self.initializer, name="arg_char_lookup")

        emb, lxl = self.arg_embedding_layer(self.char_inputs, self.tri_loc_inputs, self.arg_loc_inputs, self.char_lookup, self.trigger_sen)
        conv = self.arg_convolution_layer(emb)
        pool = self.arg_dynamic_layer(conv, self.masks)
        arg_pre = self.arg_output_layer(lxl, pool)
        self.loss = self.arg_loss_layer(self.argrole_inputs, arg_pre)
        self.train_step = tf.train.AdadeltaOptimizer(rho=0.95, epsilon=1e-6).minimize(self.loss)

    def run_step(self, sess, batch):
        chars, ls, tri, arg, arg_in, tri_loc, arg_loc, mask, cut = batch
        feed_dict = {
            self.keep_prob: 0.5,
            self.masks: mask,
            self.char_inputs: np.asarray(chars),
            self.trigger_sen: np.asarray(tri),
            self.argrole_inputs: np.asarray(arg_in),
            self.tri_loc_inputs: np.asarray(tri_loc),
            self.arg_loc_inputs: np.asarray(arg_loc)
        }
        global_step, loss, _ = sess.run([self.global_step, self.loss, self.train_step], feed_dict)
        return global_step, loss

    def arg_embedding_layer(self, char_inputs, tri_locs, arg_locs, char_lookup, tri_sen):
        with tf.variable_scope("arg_embedding", reuse=tf.AUTO_REUSE):
            # char_lookup: [20136, char_dim] char_inputs: [batch_size, sen_len]
            cwf = tf.nn.embedding_lookup(char_lookup, char_inputs)                         # [batch, sen_len, char_dim]
            pf_lookup_1 = tf.get_variable(name="arg_pf_lookup_1", shape=[2*self.sen_len-1, self.pf_dim], dtype=tf.float32, initializer=self.initializer)
            pf_lookup_2 = tf.get_variable(name="arg_pf_lookup_2", shape=[2*self.sen_len-1, self.pf_dim], dtype=tf.float32, initializer=self.initializer)
            pf_1 = tf.nn.embedding_lookup(pf_lookup_1, tri_locs)
            pf_2 = tf.nn.embedding_lookup(pf_lookup_2, arg_locs)
            ef_lookup = tf.get_variable(name="arg_ef_lookup", shape=[self.trigger_num, self.ef_dim], dtype=tf.float32, initializer=self.initializer)
            ef = tf.nn.embedding_lookup(ef_lookup, tri_sen)
            embed = tf.concat([cwf, pf_1, pf_2, ef], axis=-1)                                   # [batch, sen_len, char_dim+2*pf_dim+ef_dim]
            lxl = tf.reshape(cwf, shape=[self.batch, self.sen_len*self.char_dim])               # [batch, sen_len*char_dim]
            print("emb!")
            return embed, lxl

    def arg_convolution_layer(self, emb):
        with tf.variable_scope("arg_convolution", reuse=tf.AUTO_REUSE):                         # emb: [batch, sentence_length, char_dim+2*pf_dim+ef_dim]
            w = tf.get_variable(name="arg_conv_w", shape=[self.window, self.char_dim + 2*self.pf_dim + self.ef_dim, self.feature], dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv1d(emb, w, stride=1, padding="VALID")                              # [batch, sen_len-window+1, feature]
            b = tf.get_variable(name="arg_conv_b", shape=[self.sen_len-self.window+1, self.feature], dtype=tf.float32, initializer=self.initializer)
            conved = tf.add(tf.nn.tanh(conv), b)                                                # [batch, sen_len-window+1, feature]
            print("conved!")
            return conved

    def arg_dynamic_layer(self, conv, mask):
        with tf.variable_scope("arg_pool", reuse=tf.AUTO_REUSE):
            mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)    # [4, 3]
            mask = tf.nn.embedding_lookup(mask_embedding, mask)                                             # [batch, sen_len-window+1, 3]
            pooled = tf.reduce_max(tf.expand_dims(mask*100, 2) + tf.expand_dims(conv, 3), axis=1) - 100
            print("dynamic!")
            return tf.reshape(pooled, [-1, 3*self.feature])

    def arg_output_layer(self, lxl, pool):
        with tf.variable_scope("arg_output", reuse=tf.AUTO_REUSE):
            f = tf.concat([lxl, pool], axis=-1)                                  # [batch, sen_len*char_dim + 3*feature]
            W_s = tf.get_variable(name="arg_out_w", shape=[self.sen_len*self.char_dim + 3*self.feature, self.argrole_num], dtype=tf.float32, initializer=self.initializer)
            b_s = tf.get_variable(name="arg_out_b", shape=[self.argrole_num], dtype=tf.float32, initializer=self.initializer)
            output = tf.nn.bias_add(tf.matmul(f, W_s), b_s)                      # [batch, argrole_num]
            print("output!")
            return output

    def arg_loss_layer(self, argrole_inputs, arg_pre):
        with tf.variable_scope("arg_loss", reuse=tf.AUTO_REUSE):
            argrole_inputs = tf.cast(argrole_inputs, tf.float32)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=arg_pre, labels=argrole_inputs, name="loss"))
            print("loss!")
            return loss