import numpy as np
import os
import re
import tensorflow as tf
import matplotlib.pyplot as plt

class CNN(object):
    """文本分类，CNN模型"""

    def __init__(self, sess,
                 embedding_dim=7,  # 向量维度
                 num_steps = 30,  # 序列长度
                 num_filters = 256,  # 卷积核数目
                 kernel_size = 3,  # 卷积核尺寸
                 hidden_dim = 128,  # 全连接层神经元
                 keep_prob = 0.5,  # dropout保留比例
                 logs_dir="logs",  # 日志所在的路径（根目录）
                 plots_dir="images"
                 ):

        self.sess= sess
        self.embedding_dim = embedding_dim
        self.num_steps2 = num_steps
        self.num_filters = num_filters  # 卷积核数目
        self.kernel_size = kernel_size  # 卷积核尺寸
        self.hidden_dim = hidden_dim   # 全连接层神经元
        self.keep_prob = keep_prob # dropout保留比例
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.num_classes = 5,  # 类别数
        self.print_per_batch = 100,  # 每多少轮输出一次结果
        self.save_per_batch = 10,  # 每多少轮存入tensorboard

        self.build_graph()

    def build_graph(self):
        """CNN模型"""
        # 三个待输入的数据
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps2, self.embedding_dim], name='inputs')
        self.targets =tf.placeholder(tf.float32, [None, 5], name='targets')


        # CNN layer
        conv = tf.layers.conv1d(self.inputs, self.num_filters, self.kernel_size, name='conv')
        # global max pooling layer
        gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')


        # 全连接层，后面接dropout
        fc = tf.layers.dense(gmp, self.hidden_dim, name='fc1')
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)

        # 预测器
        self.pred = tf.layers.dense(fc, 5, name='fc2')

        # 损失函数
        self.loss = tf.reduce_mean(tf.square(self.pred-self.targets))
        # 优化器
        self.optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        #状态的保存
        self.saver = tf.train.Saver(max_to_keep=30)


    def test(self,dataset_list):
        merged_test_X = []
        merged_test_X += list(dataset_list.predict_X)
        merged_test_X = np.array(merged_test_X)  # 将测试数据集从list模式转换成矩阵 842*30*1
        test_data_feed = {
            self.learning_rate: 0.0,
            self.inputs: merged_test_X,
        }
        test_pred = self.sess.run(self.pred, test_data_feed)
        print("CNN对未来五天的预测",test_pred[-1])
        return test_pred[-1]


    def train(self,dataset_list,config):

        print("Loading training and validation data...")
        if not config.train:
            tf.global_variables_initializer().run()       #对参数进行初始化

        # 载入训练集与验证集
        merged_test_X = []
        merged_test_y = []

        # 从数据集中选出测试数据集
        merged_test_X += list(dataset_list.test_X)
        merged_test_y += list(dataset_list.test_y)

        merged_test_X = np.array(merged_test_X)  # 将测试数据集从list模式转换成矩阵 842*30*1
        merged_test_y = np.array(merged_test_y)

        test_data_feed = {
            self.learning_rate : 0.0 ,
            self.inputs : merged_test_X ,
            self.targets : merged_test_y,
        }

        global_step = 0

        print("开始训练这支股票")
        for epoch in range(config.max_epoch):  # 开始迭代，迭代的次数为max_epoch（50）
            epoch_step = 0
            learning_rate = config.init_learning_rate * (  # 对学习率进行衰减
                    config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for batch_X,batch_y in dataset_list.generate_one_epoch(config.batch_size):
                global_step += 1
                epoch_step += 1

                train_data_feed = {
                    self.learning_rate : learning_rate,
                    self.inputs : batch_X,
                    self.targets : batch_y,
                }
                train_loss,_ = self.sess.run([self.loss,self.optim],train_data_feed)    #计算损失函数，并进行权值的更新
                if np.mod(global_step,100) == 1:                                        #每训练100个数据块就进行一次测试
                    test_loss, test_pred = self.sess.run([self.loss,self.pred],test_data_feed)

                    print("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f" % (
                    # 输出当前的训练集的损失，测试集的损失
                        global_step, epoch, learning_rate, train_loss, test_loss))

                    # print("Step:%d [Epoch:%d] [Learning rate: %.6f]  test_loss:%.6f" % (
                    # # 输出当前的训练集的损失，测试集的损失
                    #     global_step, epoch, learning_rate,  test_loss))

                    # for sample_sym, indices in sample_indices.items():  # 将字典转换为列表
                    image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(  # 给图片命名
                        dataset_list.stock_sym, epoch, epoch_step))


                    indices = np.array([
                        i for i in range(len(merged_test_y))
                    ])

                    sample_preds = test_pred[indices]
                    sample_truth = merged_test_y[indices]
                    self.plot_samples(sample_preds, sample_truth, image_path,
                                      stock_sym=dataset_list.stock_sym)  # 绘图，通过预测值和真实值来绘图
                    self.save(global_step)






    @property  #可以将一个对象的方法变成它的属性
    def model_name(self):       #给神经网络起名字。。。
        name = "stock_cnn%d_step%d_input%d" % (
            self.num_filters, self.num_steps2, self.embedding_dim)

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
        num_point = len(targets)                                                          #每幅图绘制的点的个数
        truths = flatten(targets)
        preds = flatten(preds)
        days = range(len(truths))[:num_point+4]



        truths = truths[:4] + [truths[5*i+4] for i in range(num_point)]
        preds = preds[:4] + [preds[5 * i+4] for i in range(num_point)]


        plt.figure(figsize=(12, 6))
        # plt.plot(days, truths, label='truth')
        # plt.plot(days, preds, label='pred')
        # plt.bar(days, preds, width=1, color="yellow")             #绘制柱状图

        no= 0.0
        yes=0.0
        # print(preds)
        # print(truths)
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


