

import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


class NeuralNet(object):
    def __init__(self, hp):
        self.hp = hp
        self.W = np.zeros((self.hp.input_size, self.hp.output_size))
        self.B = np.zeros((1, self.hp.output_size))

    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        return Z

    def __backwardBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB

    def __update(self, dW, dB):
        self.W = self.W - self.hp.eta * dW
        self.B = self.B - self.hp.eta * dB

    def inference(self, x):
        return self.__forwardBatch(x)

    def train(self, dataReader, checkpoint=0.1):
        # calculate loss to decide the stop condition
        loss_history = TrainingHistory()
        loss = 10
        if self.hp.batch_size == -1:
            self.hp.batch_size = dataReader.num_train
        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = (int)(max_iteration * checkpoint)

        for epoch in range(self.hp.max_epoch):
            #print("epoch=%d" %epoch)
            # dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(
                    self.hp.batch_size, iteration)
                # get z from x,y
                batch_z = self.__forwardBatch(batch_x)
                # calculate gradient of w and b
                dW, dB = self.__backwardBatch(batch_x, batch_y, batch_z)
                # update w,b
                self.__update(dW, dB)

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration+1) % checkpoint_iteration == 0:
                    loss = self.__checkLoss(dataReader)
                    loss_history.AddLossHistory(
                        epoch*max_iteration+iteration, loss)
                    if loss < self.hp.eps:
                        break
                    # end if
                # end if
            # end for
            if loss < self.hp.eps:
                break
            if epoch % 100 == 0:
                print("epoch: ", epoch,
                      "loss: ", loss, "W: ", self.W, "B: ", self.B)
        # end for
        loss_history.ShowLossHistory(self.hp)

    def __checkLoss(self, dataReader):
        X, Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forwardBatch(X)
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/m/2
        return loss


class TrainingHistory(object):
    def __init__(self):
        self.iteration = []
        self.loss_history = []

    def AddLossHistory(self, iteration, loss):
        self.iteration.append(iteration)
        self.loss_history.append(loss)

    # 训练loss可视化
    def ShowLossHistory(self, hp, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(self.iteration, self.loss_history)
        title = hp.toString()
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        if xmin != None and ymin != None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()
        return title


class DataReader(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = None  # normalized x, if not normalized, same as YRaw
        self.YTrain = None  # normalized y, if not normalized, same as YRaw
        self.XRaw = None    # raw x
        self.YRaw = None    # raw y

    # 读入样本csv
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.genfromtxt(
                train_file, delimiter=",", skip_header=1)
            self.XRaw = data[:, :-1].copy()
            self.YRaw = data[:, -1].copy().reshape(len(data[:, -1]), 1)
            self.num_train = self.XRaw.shape[0]
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
            # 源数据可视化分析
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.XRaw[:, 0], self.XRaw[:, 1],
                       self.YRaw, label='Raw Data')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title(
                "It can be seen from the figure that\nthere is an obvious linear relationship between x,y and z")
            ax.legend()
            plt.show()
        else:
            raise Exception("Cannot find train file!!!")

    # 标准化样本数据
    # return: X_new: normalized data with same shape
    # return: X_norm: N x 2
    #               [[min1, range1]
    #                [min2, range2]
    #                [min3, range3]]
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros((num_feature, 2))
        # 按列归一化,即所有样本的同一特征值分别做归一化
        for i in range(num_feature):
            col_i = self.XRaw[:, i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            self.X_norm[i, 0] = min_value
            self.X_norm[i, 1] = max_value - min_value
            new_col = (col_i - self.X_norm[i, 0])/(self.X_norm[i, 1])
            X_new[:, i] = new_col
        self.XTrain = X_new

    # get batch training data
    def GetSingleTrainSample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x, y

    # get batch training data
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end, :]
        batch_Y = self.YTrain[start:end, :]
        return batch_X, batch_Y

    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain


class HyperParameters(object):
    def __init__(self, input_size, output_size, eta=0.1, max_epoch=1000, batch_size=5, eps=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps

    # 训练loss可视化表标题
    def toString(self):
        title = str.format("bz:{0},eta:{1}", self.batch_size, self.eta)
        return title


# 还原参数值
def DeNormalizeWeightsBias(net, dataReader):
    W_true = np.zeros_like(net.W)
    for i in range(W_true.shape[0]):
        W_true[i, 0] = net.W[i, 0] / dataReader.X_norm[i, 1]
    # end for
    B_true = net.B - W_true[0, 0] * dataReader.X_norm[0,
                                                      0] - W_true[1, 0] * dataReader.X_norm[1, 0]
    return W_true, B_true


if __name__ == '__main__':
    # 读入样本并标准化
    reader = DataReader("./Dataset/mlm.csv")
    reader.ReadData()
    reader.NormalizeX()
    # 神经网络初始化并训练
    hp = HyperParameters(
        2, 1, eta=0.001, max_epoch=3000, batch_size=10, eps=1.58)
    net = NeuralNet(hp)
    net.train(reader, checkpoint=0.1)
    # 还原参数值
    W_true, B_true = DeNormalizeWeightsBias(net, reader)
    print("Final linear regression model:")
    result = f"z = {W_true[0,0]} x + {W_true[1,0]} y + {B_true[0,0]}"
    print(result)
    # 结果可视化
    predictedY = np.dot(reader.XRaw, W_true)+B_true
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reader.XRaw[:, 0], reader.XRaw[:, 1],
               reader.YRaw, label='Raw Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(result)
    p = np.linspace(0, 100)
    q = np.linspace(0, 100)
    P, Q = np.meshgrid(p, q)
    R = np.hstack((P.ravel().reshape(2500, 1), Q.ravel().reshape(2500, 1)))
    Z = np.dot(R, W_true) + B_true
    Z = Z.reshape(50, 50)
    ax.plot_surface(P, Q, Z, cmap='rainbow')
    ax.legend()
    plt.show()
