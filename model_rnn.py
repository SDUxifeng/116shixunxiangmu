"""
@author: lilianweng
"""
import numpy as np
import os
import random
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt


class LstmRNN(object):
    def __init__(self, sess, stock_count,
                 lstm_size=128,       #单层神经元的个数
                 num_layers=1,        #隐含层的层数
                 num_steps=30,
                 input_size=1,        #即batchsize
                 keep_prob=0.8,       #cell的drop概率
                 logs_dir="logs",     #日志所在的路径（根目录）
                 plots_dir="images"): #图片所在的路径（根目录）
        """
        Construct a RNN model using LSTM cell.

        Args:
            sess:
            stock_count:
            lstm_size:
            num_layers
            num_steps:
            input_size:
            keep_prob:
            checkpoint_dir
        """
        self.sess = sess
        self.stock_count = stock_count
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.keep_prob = keep_prob
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.build_graph()

    def build_graph(self):
        """
        The model asks for three things to be trained:
        - input: training data X
        - targets: training label y
        - learning_rate:
        """
        # inputs.shape = (number of examples, number of input, dimension of each input).
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")

        # Stock symbols are mapped to integers.
        self.symbols = tf.placeholder(tf.int32, [None, 1], name='stock_labels')

        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")

        def _create_one_cell():              #创建一个cell
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            if self.keep_prob < 1.0:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)  #设置dropout防止过拟合
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(   #根据隐含层的层数来构建多层神经网络
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()


        # Run dynamic RNN
        val, state_ = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32, scope="dynamic_rnn")  #运行一遍神经网络


        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")    #gather函数用于取出tensor中某几位的量,结果是一个  batch_size*lstm_size的矩阵
        ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w") #正态分布出初始权值
        self.ws =ws
        bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")  #初始化偏置，值为0.1
        self.bias=bias
        self.pred = tf.matmul(last, self.ws) + self.bias                                   #生成预测结果


        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("w", self.ws)
        self.b_sum = tf.summary.histogram("b", self.bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse")                        #计算损失函数
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")    #反向传播来进行权值的更新

        self.loss_sum = tf.summary.scalar("loss_mse", self.loss)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

        self.t_vars = tf.trainable_variables()       #所有需要训练的变量的集合
        self.saver = tf.train.Saver()                #用于保存和恢复所有的变量

    def tset2(self,dataset_list,config):
        """
              Args:
                  dataset_list (<StockDataSet>)
                  config (tf.app.flags.FLAGS)
              """
        assert len(dataset_list) > 0  # 断言股票的个数不为0

        # Merged test data of different stocks.
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []


        for label_, d_ in enumerate(dataset_list):  # 从数据集中选出测试数据集
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        merged_test_X = np.array(merged_test_X)  # 将测试数据集从list模式转换成矩阵 842*30*1
        merged_test_y = np.array(merged_test_y)
        merged_test_labels = np.array(merged_test_labels)

        test_data_feed = {
            self.learning_rate: 0.0,
            self.inputs: merged_test_X,
            self.targets: merged_test_y,
            self.symbols: merged_test_labels,
        }

        global_step = 0
        random.seed(time.time())

        print("开始测试")  # 开始训练这支股票
        for epoch in range(2):  # 开始迭代，迭代的次数为max_epoch（50）
            # 以下代码将被删除
            test_loss, test_pred = self.sess.run([self.loss, self.pred], test_data_feed)
            print("代码第147行，打印训练前的权值", self.sess.run(self.ws)[-10:])

            print("Step:%d [Epoch:%d]  test_loss:%.6f" % (
                # 输出当前的训练集的损失，测试集的损失
                global_step, epoch, test_loss))
            # 以上代码将被删除
        print("测试结束")

    def test(self, dataset_list, config):
        """
        Args:
            dataset_list (<StockDataSet>)
            config (tf.app.flags.FLAGS)
        """
        assert len(dataset_list) > 0                #断言股票的个数不为0

        # Merged test data of different stocks.
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []

        for label_, d_ in enumerate(dataset_list):        #从数据集中选出测试数据集
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        merged_test_y=np.array(merged_test_y)

        '''这里是奇葩的预测的开始'''

        global_step = 0
        random.seed(time.time())
        my_test_pred = []
        '''这段代码就是为了将真实的结果集能够绘图'''
        sample_labels = range(min(config.sample_size, len(dataset_list)))  # 股票的数量
        sample_indices = {}
        for l in sample_labels:
            sym = dataset_list[l].stock_sym
            # print(sym)
            target_indices = np.array([
                i for i, sym_label in enumerate(merged_test_labels)
                if sym_label[0] == l])
            sample_indices[sym] = target_indices  # 建立字典
        '''这段代码就是为了将真实的结果集能够绘图'''

        print("开始测试")        # 开始测试这支股票

        for epoch in range(300):  # 开始迭代，迭代的次数为max_epoch（50）

            _test_X = [merged_test_X[epoch]]
            _test_y = [merged_test_y[epoch]]
            _test_labels = [[label_]]

            if len(_test_X[0]) > len(my_test_pred):
                for index, value in enumerate(
                        my_test_pred):  # 最新添加的数据将采用之前预测出来的结果来进行                                             #如果预测窗口的大小大于新生成的测试结果，则替换掉后半部分
                    _test_X[0][-len(my_test_pred) + index] = value[0]
            else:
                print("长度超了")
                for index, value in enumerate(my_test_pred[(len(my_test_pred) - len(_test_X[0])):]):
                    _test_X[0][index] = value[0]

            temp_test_X = np.array(_test_X)
            temp_test_y = np.array(_test_y)
            temp_test_labels = np.array(_test_labels)

            test_data_feed = {
                self.learning_rate: 0.0,
                self.inputs: temp_test_X,
                self.targets: temp_test_y,
                self.symbols: temp_test_labels,
            }

            test_loss, test_pred = self.sess.run([self.loss, self.pred], test_data_feed)
            my_test_pred.append(test_pred)                               #将预测的结果添加到预测结果集内
            print("Step:%d [Epoch:%d]  test_loss:%.6f" % (
                # 输出当前的训练集的损失，测试集的损失
                global_step, epoch, test_loss))


            if epoch % 110 == 109:
                sample_preds = my_test_pred
                for sample_sym, indices in sample_indices.items():  # 将字典转换为列表
                    image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(  # 给图片命名
                        sample_sym, epoch, epoch))
                    sample_truth = merged_test_y[indices]
                    print(sample_truth,"交界处")
                    print(sample_preds,"结束点")
                    self.plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym)  # 绘图，通过预测值和真实值来绘图
        print("测试结束")

        '''这里是奇葩的预测的结束'''
 


    def train(self, dataset_list, config):
        """
        Args:
            dataset_list (<StockDataSet>)
            config (tf.app.flags.FLAGS)
        """
        assert len(dataset_list) > 0                #断言股票的个数不为0
        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))

        self.writer.add_graph(self.sess.graph)
        if not config.train:
            tf.global_variables_initializer().run()

        # Merged test data of different stocks.
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []

        for label_, d_ in enumerate(dataset_list):        #从数据集中选出测试数据集
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        merged_test_X = np.array(merged_test_X)           #将测试数据集从list模式转换成矩阵 842*30*1
        merged_test_y = np.array(merged_test_y)
        merged_test_labels = np.array(merged_test_labels)

        print ("len(merged_test_X) =", len(merged_test_X))        #打印输出数据集内数据的数量
        print ("len(merged_test_y) =", len(merged_test_y))
        print ("len(merged_test_labels) =", len(merged_test_labels))

        test_data_feed = {
            self.learning_rate: 0.0,
            self.inputs: merged_test_X,
            self.targets: merged_test_y,
            self.symbols: merged_test_labels,
        }

        global_step = 0

        num_batches = sum(len(d_.train_X) for d_ in dataset_list) // config.batch_size
        random.seed(time.time())

        # Select samples for plotting.
        sample_labels = range(min(config.sample_size, len(dataset_list)))    #股票的数量
        sample_indices = {}
        for l in sample_labels:
            sym = dataset_list[l].stock_sym
            # print(sym)
            target_indices = np.array([
                i for i, sym_label in enumerate(merged_test_labels)
                if sym_label[0] == l])
            sample_indices[sym] = target_indices                             #建立字典
        # print (sample_indices)

        print ("Start training for stocks:", [d.stock_sym for d in dataset_list])   #开始训练这支股票
        for epoch in range(config.max_epoch):                                       #开始迭代，迭代的次数为max_epoch（50）
            epoch_step = 0
            learning_rate = config.init_learning_rate * (                           #对学习率进行衰减
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for label_, d_ in enumerate(dataset_list):                               #对每支股票进行迭代
                for batch_X, batch_y in d_.generate_one_epoch(config.batch_size):           #对数据集进行预处理，并开始迭代
                    global_step += 1
                    epoch_step += 1
                    batch_labels = np.array([[label_]] * len(batch_X))                      #这是什么啊？股票的标记？干什么用的？
                    train_data_feed = {                                                     #对预定义的内容进行填充，分别是学习率，训练数据集，训练标签集，股票的标记
                        self.learning_rate: learning_rate,
                        self.inputs: batch_X,
                        self.targets: batch_y,
                        self.symbols: batch_labels,
                    }

                    train_loss, _, train_merged_sum = self.sess.run(                        #计算损失函数，进行权值更新，还有一个是用于可视化网络结构和参数
                        [self.loss, self.optim, self.merged_sum], train_data_feed)
                    self.writer.add_summary(train_merged_sum, global_step=global_step)

                    if np.mod(global_step, len(dataset_list) * 100 / config.input_size) == 1:                          #每训练100个数据块就进行一次测试
                        test_loss, test_pred = self.sess.run([self.loss, self.pred], test_data_feed)

                        print ("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f" % (           #输出当前的训练集的损失，测试集的损失
                            global_step, epoch, learning_rate, train_loss, test_loss))


                        # Plot samples
                        print("clllld")
                        for sample_sym, indices in sample_indices.items():                                             #将字典转换为列表
                            image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(    #给图片命名
                                sample_sym, epoch, epoch_step))
                            print("cnnnd")
                            sample_preds = test_pred[indices]
                            sample_truth = merged_test_y[indices]
                            self.plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym)            #绘图，通过预测值和真实值来绘图

                        self.save(global_step)

        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)                                 #最终的预测以及最终的损失
        print("训练结束，最终的test_loss为" , global_step, epoch,test_loss)                                            # 输出最终的测试集的损失
        print("最终的W为",self.ws)

        # Save the final model
        self.save(global_step)
        return final_pred

    @property  #可以将一个对象的方法变成它的属性
    def model_name(self):       #给神经网络起名字。。。
        name = "stock_rnn_lstm%d_step%d_input%d" % (
            self.lstm_size, self.num_steps, self.input_size)

        return name

    @property
    def model_logs_dir(self):   #日志所在的路径，如果不存在就创建该路径（子目录）
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir

    @property
    def model_plots_dir(self):  #图片所在的路径，如果不存在就创建该路径（子目录）
        model_plots_dir = os.path.join(self.plots_dir, self.model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir

    def save(self, step):       #保存当前模型
        model_name = self.model_name + ".model"
        self.saver.save(
            self.sess,
            os.path.join(self.model_logs_dir, model_name),
            global_step=step
        )

    def load(self):             #加载模型，并恢复到最新保存的那一次

        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)   #打开checkpoint文件，通过该文件来对保存的模型进行操作
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))     #模型状态的回复，新状态保存到了sess内，这是个面向过程的方法。。。
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))

            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



    def plot_samples(self, preds, targets, figname, stock_sym=None):             #绘图，将当前的预测值和真实值绘制成png
        def flatten(seq):  # 将二维的数组变成一维的
            return [x for y in seq for x in y]
        num_point = 100                                                           #每幅图绘制的点的个数
        truths = flatten(targets)[:num_point]
        preds = flatten(preds)[:num_point]
        days = range(len(truths))[:num_point]


        plt.figure(figsize=(12, 6))
        # plt.plot(days, truths, label='truth')
        # plt.plot(days, preds, label='pred')
        # plt.bar(days, preds, width=1, color="yellow")             #绘制柱状图

        no= 0.0
        yes=0.0
        for index in range(len(preds)):
            if preds[index]*truths[index]<0:
                no +=1
            else:
                yes +=1
        rate = yes/(no+yes)
        print("yes:%d,no%d,rate%d:",yes,no,rate)

        truth1 = list(truths)
        truth2 = list(truths)
        for index , value in enumerate(truths):
            if value >0:
                truth1[index] = 0
            else:
                truth2[index] = 0

        plt.bar(days,truth1,width=1,color="green")
        plt.bar(days,truth2,width=1,color="red")
        plt.plot(days,preds,label = "pred")


        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("day")
        plt.ylabel("normalized price")
        plt.ylim((min(truths), max(truths)))
        plt.grid(ls='--')

        if stock_sym:
            plt.title(stock_sym + " | Last %d days in test" % len(truths))

        plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
        plt.close()
