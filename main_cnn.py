import os
import pandas as pd
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_model import StockDataSet
from model_cnn import CNN

flags = tf.app.flags        #程序在命令行运行时可以输入参数
flags.DEFINE_integer("embedding_dim", 7, "Input size [1]")#input data sample                          #每个窗口新添加的数据的大小
flags.DEFINE_integer("num_steps", 30, "Num of steps [30]")                                          #每次训练、预测的窗口的大小
flags.DEFINE_integer("kernel_size", 3, "Num of layer [1]")                                           #卷积核的尺寸
flags.DEFINE_integer("num_filters", 256, "Size of one LSTM cell [128]")                              #卷积核的个数
flags.DEFINE_integer("hidden_dim", 128, "Size of one LSTM cell [128]")                               #全连接层神经元的个数
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")                             #对训练数据切割成多组，每组的个数
flags.DEFINE_float("keep_prob", 0.9, "Keep probability of dropout layer. [0.8]")                    #LSTM网络内drop的概率，0代表完全不通过，1代表完全通过
flags.DEFINE_float("init_learning_rate", 0.002, "Initial learning rate at early stage. [0.001]")    #初始化学习率
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")              #学习率的衰减率
flags.DEFINE_integer("max_epoch", 500, "Total training epoches. [50]")                               #迭代的次数
flags.DEFINE_string("stock_symbol", "szzs", "Target stock symbol [None]")                            #股票的名字，它仿佛是想做一个队列？但是这为啥就只能填一个
flags.DEFINE_boolean("train",False, "True for training, False for testing [False]")                #一个变量来决定神经网络是否使用之前训练好的权值
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()   #用于打印输出，pprint模块 提供了打印出任何Python数据结构类和方法。

if not os.path.exists("logs"):
    os.mkdir("logs")

def load_sp500(input_size, num_steps,target_symbol=None, test_ratio=0.05):
    if target_symbol is not None:
        return StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio)

def main(_):
    pp.pprint(FLAGS.__flags)
    run_config = tf.ConfigProto()                #控制GPU资源的使用
    run_config.gpu_options.allow_growth = True    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        cnn_model = CNN(
            sess,
            embedding_dim = FLAGS.embedding_dim,
            num_steps=FLAGS.num_steps,
            num_filters=FLAGS.num_filters,  # 卷积核数目
            kernel_size=FLAGS.kernel_size,  # 卷积核尺寸
            hidden_dim=FLAGS.hidden_dim,  # 全连接层神经元
            keep_prob=FLAGS.keep_prob,
        )

        stock_data_list = load_sp500(
            FLAGS.embedding_dim,
            FLAGS.num_steps,
            target_symbol=FLAGS.stock_symbol,
        )


        if FLAGS.train:
            cnn_model.load()
        cnn_model.train(stock_data_list, FLAGS)
        # rnn_model.test(stock_data_list, FLAGS)
        # else:
        #    if not rnn_model.load()[0]:
        #         raise Exception ("[!] Train a model first, then run test mode")



if __name__ == '__main__':
    print(1)
    print(2)
    tf.app.run()
