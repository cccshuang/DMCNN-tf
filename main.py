import os
import pickle
import numpy as np
import tensorflow as tf

from batch import load_tri_sentences, load_arg_sentences, Batch_tri, Batch_arg
from model_tri import create_model_tri
from model_arg import create_model_arg


flags = tf.flags
flags.DEFINE_string("gpu",              "4",                "")
flags.DEFINE_string("tri_path",         'data/tri.train',   "")
flags.DEFINE_string("arg_path",         'data/arg.train',   "")
flags.DEFINE_integer("train_epoch",     1000,                "")
flags.DEFINE_integer("sen_len",         80,                 "")
flags.DEFINE_integer("char_dim",        100,                "")
flags.DEFINE_integer("num_char",        20136,              "")
flags.DEFINE_integer("tri_batch",       170,                "")
flags.DEFINE_integer("tri_num",         34,                 "")
flags.DEFINE_integer("tri_pf_dim",      5,                  "")
flags.DEFINE_integer("tri_window",      3,                  "")
flags.DEFINE_integer("tri_feature",     200,                "")
flags.DEFINE_integer("arg_batch",       20,                 "")
flags.DEFINE_integer("arg_num",         36,                 "")
flags.DEFINE_integer("arg_pf_dim",      5,                  "")
flags.DEFINE_integer("arg_ef_dim",      5,                  "")
flags.DEFINE_integer("arg_window",      3,                  "")
flags.DEFINE_integer("arg_feature",     300,                "")
FLAGS = flags.FLAGS

def load_config():
    config = dict()
    config["sen_len"] = FLAGS.sen_len
    config["char_dim"] = FLAGS.char_dim
    config["num_char"] = FLAGS.num_char
    config["tri_batch"] = FLAGS.tri_batch
    config["tri_num"] = FLAGS.tri_num
    config["tri_pf_dim"] = FLAGS.tri_pf_dim
    config["tri_window"] = FLAGS.tri_window
    config["tri_feature"] = FLAGS.tri_feature
    config["arg_batch"] = FLAGS.arg_batch
    config["arg_num"] = FLAGS.arg_num
    config["arg_pf_dim"] = FLAGS.arg_pf_dim
    config["arg_ef_dim"] = FLAGS.arg_ef_dim
    config["arg_window"] = FLAGS.arg_window
    config["arg_feature"] = FLAGS.arg_feature
    return config

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    train_tri = load_tri_sentences(FLAGS.tri_path)
    train_tri_b = Batch_tri(train_tri, FLAGS.tri_batch, FLAGS.sen_len)
    data = train_tri_b.batch_data
    with open("data/maps.pkl", 'rb') as f:
        _, id_to_char, __, ___ = pickle.load(f)
    config = load_config()
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session() as sess:
        model_tri = create_model_tri(sess, config, id_to_char)
        loss = list()
        for i in range(FLAGS.train_epoch):
            for batch in train_tri_b.iter_batch():
                step, batch_loss, max = model_tri.run_step(sess, batch)
                loss.append(batch_loss)
                print(max)
            loss_av = tf.reduce_mean(tf.convert_to_tensor(loss))
            loss = []
            print(str(i) + ':  ' + str(sess.run(loss_av)))
    # train_arg = load_arg_sentences(FLAGS.arg_path)
    # train_arg_b = Batch_arg(train_arg, FLAGS.arg_batch, FLAGS.sen_len)
    # data = train_arg_b.batch_data
    # with open("data/maps.pkl", 'rb') as f:
    #     _, id_to_char, __, ___ = pickle.load(f)
    #     config = load_config()
    #     # tf_config = tf.ConfigProto()
    #     # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #     with tf.Session() as sess:
    #         model_arg = create_model_arg(sess, config, id_to_char)
    #         loss = list()
    #         for i in range(FLAGS.train_epoch):
    #             for batch in train_arg_b.iter_batch():
    #                 step, batch_loss = model_arg.run_step(sess, batch)
    #                 loss.append(batch_loss)
    #             loss_av = tf.reduce_mean(tf.convert_to_tensor(loss))
    #             loss = []
    #             print(str(i) + ':  ' + str(sess.run(loss_av)))


