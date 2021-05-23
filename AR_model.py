import numpy as np 
import scipy as sp 
from scipy import signal as sig
from matplotlib import pyplot as plt
import math

def LevinsonDurbin(r, lpcOrder):
    '''
    r:自相关函数向量，长度为2N-1，N为采样点数
    lpcOrder:LevinsonDrubin递推阶数，即选用几个x_n前几项来递推

    return:
    a:m阶递推系数
    rou[-1]:m阶的误差的期望
    '''
    #a用来存放系数
    a = np.zeros(lpcOrder + 1, dtype=np.float64)
    #rou用来存放误差的期望rou = E(e_n^2)
    rou = np.zeros(lpcOrder + 1, dtype=np.float64)

    #确定递推初始值
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    rou[1] = r[0] + r[1] * a[1]
    #lam是反射系数
    lam = - r[1] / r[0]

    for m in range(1, lpcOrder):

        #反射系数km， m阶的第m个系数a_m(m) = R_m
        lam = 0.0

        #系数a的迭代方程
        for i in range(m + 1):
            #这里的lam是系数和自相关系数R的累加和
            lam -= a[i] * r[m + 1 - i]

        lam /= rou[m]

        U = [1]
        U.extend([a[j] for j in range(1, m + 1)])
        U.append(0)

        V = [0]
        V.extend([a[j] for j in range(m, 0, -1)])
        V.append(1)

        #这里a是一个一维数组，每次迭代之后，上一次的a都被替换
        #这里的a最终保留的是m阶的系数
        a = np.array(U) + lam * np.array(V)
        rou[m + 1] = rou[m] * (1.0 - lam * lam)

    return a, rou[-1]

class SDAR(object):
    def __init__(self, R, order):
        #序列中的一个值
        self.R = R
        #概率密度函数均值
        self.mu = np.random.random()
        #概率密度函数协方差矩阵
        self.sigma = np.random.random()
        #AR模型阶数
        self.order = order
        #各位变量的均值，c[0]位用来存储
        self.c = np.random.random(self.order+1) / 100.0

    def update(self, x, term):
        '''
        更新打分值， 和预测值
        '''
        assert len(term) >= self.order, "term must be order or more"
        #时间序列
        term = np.array(term)
        #mu用来存放当前的均值
        self.mu = (1.0 - self.R) * self.mu + self.R * x

        #列表c用来存放每一阶的均值
        for i in range(1, self.order+1):
            self.c[i] = (1 - self.R) * self.c[i] + self.R * (x - self.mu) * (term[-i] - self.mu)
        self.c[0] = (1 - self.R) * self.c[0] + self.R * (x - self.mu) ** 2

        #递推算法
        omega, e = LevinsonDurbin(self.c, self.order)

        #矩阵相乘，结果为一个序列，累加后为预测值
        weighted_seq = np.dot(-omega[1:], (term[::-1] - self.mu)) + self.mu

        self.sigma = (1 - self.R) * self.sigma + self.R * (x - weighted_seq) *(x - weighted_seq)

        #返回概率密度函数
        
        return (-math.log(math.exp(-0.5*(x-weighted_seq)**2/self.sigma)/((2 * math.pi)**0.5*self.sigma**0.5)))*(weighted_seq**self.order), weighted_seq

def raw_signal(n, f1, f2, sigma):
    '''
    创建原始信号，有两个频率分量

    n: 采样的第n个点
    sigma:加入的白噪声的方差

    return:
    采样后的信号
    '''
    if n<1 or n>1024:
        return 0
    else:
        sig1 = 1 * np.sin(2 * np.pi * f1 * n + np.pi / 3)
        sig2 = 10 * np.sin(1 * np.pi * f2 * n + np.pi / 4)
        white_gain_noise = np.random.normal(0, sigma, 1)[0]
        signal = sig1 + sig2 + white_gain_noise
        return signal

def signal(N, f1, f2, sigma=1, start=1):
    '''
    生成采样信号

    N:采样点数
    f1:频率分量1
    f2:频率分量2
    sigma：白噪声方差
    start：采样的起始点

    return：采样后的信号，存放在列表里
    '''

    data = []
    for i in range(start, start+N):
        data.append(raw_signal(i, f1, f2, sigma))
    return data
#求自相关
def xcorr(data):
    length = len(data)
    Rx = []
    data_reverse = data[::-1]
    Rx = sig.convolve(data, data_reverse) / length
    return Rx



if __name__ == '__main__':
    f1 = 0.1
    f2 = 0.2
    #生成信号

    p = 9
    N = 32
    #生成信号
    data = signal(N, f1, f2, start=64)
    #求自相关
    Rx1 = xcorr(data)
    Rx = Rx1[N-1:N+p]
    a, e = LevinsonDurbin(Rx, p)

    #显示预测结果
    plt.plot(data, c='r')
    sums = 0.0
    predict_data = []
    for j in range(64, 64+N+1):
        for i in range(p):
            sums += a[p-i] * raw_signal((j-i), f1, f2, 1)
        predict_data.append(-sums)
    plt.plot(predict_data, c = 'g')
    plt.show()
            

    