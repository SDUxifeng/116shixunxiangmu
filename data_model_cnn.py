import numpy as np
import os
import pandas as pd
import random
import time

random.seed(time.time())

#test_ratio:split train and test
class StockDataSet(object):
    def __init__(self,
                 stock_sym,        #股票的名字
                 input_size=7,     #从.CSV文件内读取的维度数
                 num_steps=30,     #窗口的长度
                 test_ratio=0.1,   #测试集所占的比例
                 ):
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio

        # Read csv file
        raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))   #读取csv文件
        # Merge into one sequence
        self.raw_seq = [price for tup in raw_df[['Open', 'Close','Max','Min','amount','predict','opinion']].values for price in tup]

        self.raw_seq = np.array(self.raw_seq)          #得到输入数据矩阵
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def _prepare_data(self, seq):           #对数据进行预处理
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])     #将一维的输入数据按照 每个向量的维度数目分割转换成二维的数据矩阵
               for i in range(len(seq) // self.input_size)]

                                                             #进行标准化处理  当前滑动窗口大小的数据/上一个滑动窗口的收盘价
        seq = [ np.append(
            np.append(seq[0][:4] / seq[0][0] - 1.0,seq[0][4]/1000000000)
            ,seq[0][5:])] + [
            np.append(
            np.append(curr[:4] / seq[i][-6] - 1.0, curr[4]/1000000000)
                ,curr[5:] )for i, curr in enumerate(seq[1:])]

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])  #将每一次的训练数据组成一个集合
        y = np.array([[seq[i + self.num_steps][1]] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))                                    #分割出训练集和测试集
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):                  #产生一个生成器
        num_batches = int(len(self.train_X)) // batch_size     #计算共有多少批次
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = list(range(num_batches))               #建立批次的索引
        random.shuffle(batch_indices)                          #对索引进行随机排列
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y
