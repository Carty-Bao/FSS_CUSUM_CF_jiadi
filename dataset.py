import numpy as np
import math
from numpy import random as rd 
from numpy.random import dirichlet
from matplotlib import pyplot as plt
import statsmodels.api as sm
import pandas as pd
import configparser
import scipy.stats as stats

class datasets(object):

    def __init__(self, n_samples=1000, n_bkp=10):
        self.n_samples = n_samples
        self.n_bkp = n_bkp
        self.init_PRI = 0

    def draw_bkps(self, n_samples=100, n_bkps=3):
        """Draw a random partition with specified number of samples and specified
        number of changes."""
        alpha = np.ones(n_bkps + 1) / (n_bkps + 1) * 2000
        bkps = np.cumsum(dirichlet(alpha) * n_samples).astype(int).tolist()
        bkps[-1] = n_samples
        return bkps

    def jumping_mean_random(self, n_samples=100, n_bkps=10, n_outlier=10, noise_std=1, outlier = False):
        #利用dirichlet分布随机生成间断点
        bkps = self.draw_bkps(n_samples, n_bkps)
        outlier_points = self.draw_bkps(n_samples, n_outlier)
        
        #生成signal,detail 为下标
        signal = np.empty(n_samples)
        detail = np.arange(n_samples)

        #mean value
        center = np.zeros(1)
        for seq in np.split(detail, bkps):
            if seq.size > 0:
                jump = rd.uniform(1, 10, size=1)
                spin = rd.choice([1, -1], 1)
                center += jump * spin
                signal[seq] = center

        #加入噪声
        if noise_std is not None:
            noise = rd.normal(size=signal.shape) * noise_std
            signal = signal + noise

        if outlier == True:
            for point in outlier_points:
                spin = rd.choice([np.max(signal), np.min(signal)], 1)
                signal[point-1] = spin
        return signal, bkps, outlier_points
    
    def jumping_mean(self, n_samples=100, n_bkps=10, n_outlier=10, noise_std=0.5, outlier = False):
        #利用dirichlet分布随机生成间断点
        bkps = self.draw_bkps(n_samples, n_bkps)

        #生成signal,detail 为下标
        signal = np.empty(n_samples)
        detail = np.arange(n_samples)

        #mean value
        center = np.zeros(1)
        for seq in np.split(detail, bkps):
            if seq.size > 0:
                jump = 4
                center += jump
                signal[seq] = center

        #加入噪声
        if noise_std is not None:
            noise = rd.normal(size=signal.shape) * noise_std
            signal = signal + noise

        if outlier == True:
            outlier_points = self.draw_bkps(n_samples, n_outlier)
            for point in outlier_points:
                spin = rd.choice([np.max(signal), np.min(signal)], 1)
                signal[point-1] = spin
        return signal, bkps, outlier_points

    def jumping_variance_random(self, n_samples=100, n_bkp=10):
        bkps = self.draw_bkps(n_samples, n_bkp)

        signal = np.empty(n_samples)
        detail = np.arange(n_samples)

        center = 0
        for seq in np.split(detail, bkps):
            if seq.size>0:
                var = rd.choice([1, 5], 1)
                noise = rd.normal(center, var, len(seq))
                signal[seq] = center + noise
        return signal, bkps

    def jumping_variance(self, n_samples=100, n_bkp=10):
        bkps = self.draw_bkps(n_samples, n_bkp)

        signal = np.empty(n_samples)
        detail = np.arange(n_samples)

        center = 0
        var = 1
        for seq in np.split(detail, bkps):
            if seq.size>0:
                if var == 1:
                    var = 5
                else:
                    var = 1
                noise = rd.normal(center, var, len(seq))
                signal[seq] = center + noise
        return signal, bkps
    
    def jumping_mean_variance(self, n_samples=100, n_bkp=10):
        bkps = self.draw_bkps(n_samples, n_bkp)

        signal = np.empty(n_samples)
        detail = np.arange(n_samples)

        center = 0
        var = 1
        for seq in np.split(detail, bkps):
            if seq.size>0:
                jump = 2
                var = rd.choice([1, 5], 1)
                center += jump
                noise = rd.normal(center, var, len(seq))
                signal[seq] = center + noise
        return signal, bkps

    def PRI_Gauss_Jitter(self):
        '''
        生成高斯脉冲抖动PRI脉冲，具体参数设置存放在PRI_signal.ini中
        '''
        def PRI_session(session):
            '''
            在一个session中生成对应的高斯抖动pri脉冲
            '''
            signal_para = []

            for elem in session.split(' '):
                signal_para.append(int(elem))
            
            order = signal_para[0]
            priIni = signal_para[1]
            priDev = signal_para[2]
            num = signal_para[3]

            X = stats.truncnorm((0 - priIni)/priDev, (priIni + priDev)/priDev, loc = priIni, scale = priDev)
            
            signal_session = X.rvs(num)
            return signal_session, signal_para
    
        signal = []
        cf = configparser.ConfigParser()
        cf.read("PRI_signal.ini", encoding='UTF-8')
        plt.rcParams['axes.unicode_minus'] = False

        bkp_points = int(cf.get("signal", "bkp_points"))
        sim_signal = str(cf.get("signal", "sim_signal"))
        if len(sim_signal.split('\n')) != bkp_points + 1: 
            print("session number is not match, please try to modify PRI_signal.ini \n")

        bkps = []
        mean = []
        var = []
        for session in sim_signal.split('\n'):
            signal_session, signal_para = PRI_session(session)
            mean.append(signal_para[1])
            var.append(signal_para[2])
            bkps.append(signal_para[3])
            signal.extend(signal_session)
        bkps = list(np.cumsum(bkps))
        return signal, bkps, mean, var

    def PRI_norm_Jitter(self):
        '''
        生成均匀抖动PRI脉冲
        '''
        def PRI_session(session):
            '''
            在一个session中生成对应的均匀抖动pri脉冲
            '''
            signal_para = []

            for elem in session.split(' '):
                signal_para.append(int(elem))
            
            order = signal_para[0]
            lower_bound = signal_para[1]
            upper_bound = signal_para[2]
            num = signal_para[3]

            signal_session = np.random.uniform(low=lower_bound, high=upper_bound, size=num)
            
            return signal_session, signal_para

        signal = []
        cf = configparser.ConfigParser()
        cf.read("PRI_signal.ini", encoding='UTF-8')
        plt.rcParams['axes.unicode_minus'] = False

        bkp_points = int(cf.get("signal", "bkp_points"))
        norm_signal = str(cf.get("signal", "norm_signal"))
        if len(norm_signal.split('\n')) != bkp_points + 1: 
            print("session number is not match, please try to modify PRI_signal.ini \n")

        bkps = []
        lower_bound = []
        upper_bound = []
        for session in norm_signal.split('\n'):
            signal_session, signal_para = PRI_session(session)
            lower_bound.append(signal_para[1])
            upper_bound.append(signal_para[2])
            bkps.append(signal_para[3])
            signal.extend(signal_session)
        bkps = list(np.cumsum(bkps))
        return signal, bkps, lower_bound, upper_bound

    def PRI_rayleigh_Jitter(self):
        '''
        生成锐利抖动PRI脉冲
        '''
        def PRI_session(session):
            '''
            在一个session中生成对应的锐利抖动pri脉冲
            '''
            signal_para = []

            for elem in session.split(' '):
                signal_para.append(int(elem))
            
            order = signal_para[0]
            scale = signal_para[1]
            num = signal_para[2]

            signal_session = np.random.rayleigh(scale, num)
            
            return signal_session, signal_para 

        signal = []
        cf = configparser.ConfigParser()
        cf.read("PRI_signal.ini", encoding='UTF-8')
        plt.rcParams['axes.unicode_minus'] = False

        bkp_points = int(cf.get("signal", "bkp_points"))
        norm_signal = str(cf.get("signal", "rayleigh_signal"))
        if len(norm_signal.split('\n')) != bkp_points + 1: 
            print("session number is not match, please try to modify PRI_signal.ini \n")

        bkps = []
        scale = []
        for session in norm_signal.split('\n'):
            signal_session, signal_para = PRI_session(session)
            scale.append(signal_para[1])
            bkps.append(signal_para[2])
            signal.extend(signal_session)
        bkps = list(np.cumsum(bkps))
        return signal, bkps, scale
        
    def PRI2TOA(self, PRI_signal):
        #若第一个数据为0，其他数据减去第一个
        PRI_signal_init = []
        self.init_PRI = PRI_signal[0]
        for index in range(len(PRI_signal)):
            PRI_signal_init.append(PRI_signal[index] - PRI_signal[0])
            TOA = np.cumsum(PRI_signal_init)
        return TOA

    def TOA2PRI(self, TOA_signal):
        '''
        将一个数据序列从TOA转变为PRI
        '''
        diff = [TOA_signal[i+1] - TOA_signal[i] for i in range(len(TOA_signal)-1)]
        for index in range(len(diff)):
            diff[index] += self.init_PRI
        return diff

    def add_spur_PRI(self, PRI_signal, para, mode='pulse_density'):
        '''
        添加虚假脉冲
        mode可以选择pulse_density或者pulse_ratio
        '''
        TOA_signal = self.PRI2TOA(PRI_signal)
        if mode == 'pulse_density':
            #计算虚假脉冲个数
            spur_num = math.floor(para * max(TOA_signal) / 1000000)
            #生成虚假脉冲序列
            spur_toa_seq = list(np.random.uniform(low=0, high=max(TOA_signal), size=spur_num))
            #将脉冲转换为list格式
            TOA_signal = list(TOA_signal)
            #将虚假脉冲序列拼接到脉冲序列之后
            TOA_signal.extend(spur_toa_seq)
            #按照升序排列
            TOA_signal.sort()
            #求一阶差分，转换为PRI序列
            PRI_spur = self.TOA2PRI(TOA_signal)
            
            return PRI_spur

        if mode == 'pulse_ratio':
            #计算虚假脉冲个数
            spur_num = math.floor(para * max(TOA_signal))
            spur_toa_seq = list(np.random.uniform(low=0, high=max(TOA_signal), size=spur_num))
            TOA_signal = list(TOA_signal)
            TOA_signal.extend(spur_toa_seq)
            TOA_signal.sort()
            PRI_spur = self.TOA2PRI(TOA_signal)
            return PRI_spur

    def miss_PRI(self, PRI_signal, miss_ratio):
        '''
        缺失脉冲
        miss_ratio 为 缺失脉冲的比例
        '''
        TOA_signal = list(self.PRI2TOA(PRI_signal))
        #计算缺失脉冲个数
        miss_num = math.floor(len(TOA_signal) * miss_ratio)
        #按照均匀分布选出缺失脉冲
        miss_pulse = list(np.random.uniform(low=0, high=len(TOA_signal), size=miss_num))
        #把miss_pulse取整
        for index, pulses in enumerate(miss_pulse):
            miss_pulse[index] = math.floor(pulses)
        #从TOA中丢弃
        TOA_signal_miss = [pulse for index, pulse in enumerate(TOA_signal) if index not in miss_pulse]
        PRI_signal_miss = self.TOA2PRI(TOA_signal_miss)
        return PRI_signal_miss
        
    def del_outlier(self, signal, points):
        '''
        删除缺失脉冲和虚假脉冲
        '''
        TOA_signal = self.PRI2TOA(signal)
        for point in points:
            #遇到离群点之后，采用最佳估计来代替离群点
            # TOA_signal[point] = np.mean([TOA_signal[point-1], TOA_signal[point+1]])
            TOA_signal[point] = 0
        TOA_signal = list(TOA_signal)
        for index in np.linspace(len(TOA_signal)-1, 1, len(TOA_signal)-1, dtype=int):
            if TOA_signal[index] == 0:
                del TOA_signal[index]
        PRI_signal = self.TOA2PRI(TOA_signal)
        return PRI_signal

    
            
def data_evaluate(signal):
    '''
    测试并找出ARMA模型的参数
    '''
    signal = pd.DataFrame(signal)
    diff1 = signal.diff(1)

    diff1.dropna(inplace=True)
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(diff1, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diff1, lags=40, ax=ax2)
    plt.show()

    arma_mod01 = sm.tsa.ARMA(diff1, (0, 1)).fit()
    print("ARMA_mod01: ", arma_mod01.aic, arma_mod01.bic, arma_mod01.hqic)
    arma_mod20 = sm.tsa.ARMA(diff1, (2, 0)).fit()
    print("ARMA_mod20: ", arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
    arma_mod21 = sm.tsa.ARMA(diff1, (2, 1)).fit()
    print("ARMA_mod21: ", arma_mod21.aic, arma_mod21.bic, arma_mod21.hqic)


if __name__ == '__main__':

    # dataset = datasets(1000, 10)
    # signal, bkps = dataset.jumping_mean_random(1000, 10)
    # plt.plot(signal)
    # plt.show()


    #生成高斯抖动PRI脉冲 
    # dataset = datasets()
    # signal, bkps, mean, var = dataset.PRI_Gauss_Jitter()

    #生成均匀抖动PRI脉冲
    # dataset = datasets()
    # signal, bkps, lower_bound, upper_bound = dataset.PRI_norm_Jitter()

    #生成瑞利分布PRI脉冲
    dataset = datasets()
    signal, bkps, scale = dataset.PRI_rayleigh_Jitter()
    # #在高斯抖动PRI脉冲中加入虚假脉冲
    # PRI_spur = dataset.add_spur_PRI(signal, para=0.001, mode='pulse_ratio')    #添加虚假脉冲

    #在高斯抖动PRI脉冲中删除一些脉冲
    # PRI_miss = dataset.miss_PRI(signal, miss_ratio=0.05)

    #绘图
    plt.scatter(range(len(signal)), signal, marker='+', color='b')
    # plt.ylim(20, 125)
    plt.xlabel("time/s")
    plt.ylabel("amplitude")
    plt.show()