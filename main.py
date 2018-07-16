import os
import pandas as pd
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_model import StockDataSet
from model_rnn import LstmRNN

flags = tf.app.flags        #程序在命令行运行时可以输入参数
flags.DEFINE_integer("stock_count", 100, "Stock count [100]")
flags.DEFINE_integer("input_size", 1, "Input size [1]")#input data sample                           #每个窗口新添加的数据的大小
flags.DEFINE_integer("num_steps", 31, "Num of steps [30]")                                          #每次训练、预测的窗口的大小
flags.DEFINE_integer("num_layers", 3, "Num of layer [1]")                                           #神经网络的层数
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")                               #单层神经网络内神经元的个数
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")                             #对训练数据切割成多组，每组的个数
flags.DEFINE_float("keep_prob", 0.9, "Keep probability of dropout layer. [0.8]")                    #LSTM网络内drop的概率，0代表完全不通过，1代表完全通过
flags.DEFINE_float("init_learning_rate", 0.002, "Initial learning rate at early stage. [0.001]")    #初始化学习率
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")              #学习率的衰减率
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 500, "Total training epoches. [50]")                               #迭代的次数
flags.DEFINE_string("stock_symbol", "szzs", "Target stock symbol [None]")                            #股票的名字，它仿佛是想做一个队列？但是这为啥就只能填一个
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train",True, "True for training, False for testing [False]")                #一个变量来决定神经网络是否使用之前训练好的权值

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()   #用于打印输出，pprint模块 提供了打印出任何Python数据结构类和方法。

if not os.path.exists("logs"):
    os.mkdir("logs")

def load_sp500(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05):
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio)
        ]



def main(_):
    pp.pprint(FLAGS.__flags)
    run_config = tf.ConfigProto()                #控制GPU资源的使用
    run_config.gpu_options.allow_growth = True    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        rnn_model = LstmRNN(
            sess,
            FLAGS.stock_count,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            keep_prob=FLAGS.keep_prob,
        )
        stock_data_list = load_sp500(
            FLAGS.input_size,
            FLAGS.num_steps,
            k=FLAGS.stock_count,
            target_symbol=FLAGS.stock_symbol,
        )


        if FLAGS.train:
            rnn_model.load()
        rnn_model.train(stock_data_list, FLAGS)
        # rnn_model.test(stock_data_list, FLAGS)
        # else:
        #    if not rnn_model.load()[0]:
        #         raise Exception ("[!] Train a model first, then run test mode")



if __name__ == '__main__':
    print(1)
    print(2)
    tf.app.run()
