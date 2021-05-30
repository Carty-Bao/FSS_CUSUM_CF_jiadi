# 多功能雷达切换点检测实验仿真

## 1.数据集生成 dataset.py
### 1）description：
jumping_mean：可以生成均值跳跃数据集合，数据服从高斯分布，可以自定义方差和均值，切换点自动生成，服从dirichlet分布。

jumping_mean_random:描述同上，数据跳跃方向随机，尺度随机，服从二项分布

jumping_variance：可以生成方差跳跃数据集合，数据服从高斯分布，可以自定义均值，切换点自动生成，服从dirichlet分布。

jumping_variance_random:描述同上，方差增减随机，数据服从二项分布。

jumping_variance_mean:可以生成均值方差同时跳跃的数据集合，数据服从高斯分布，切换点自动生成，服从dirichlet分布。

PRI_Gauss_Jitter：生成多功能雷达高斯抖动PRI数据，数据具体信息在配置文件PRI_signal.ini中配置。通过修改sim_signal变量和bkp_points变量来修改数据段的结构。

PRI_norm_Jitter：生成多功能雷达均匀抖动PRI数据，数据具体信息在配置文件PRI_signal.ini中配置。通过修改norm_signal变量和bkp_points变量来修改数据段的结构。

PRI_rayleigh_Jitter：生成多功能雷达瑞利抖动PRI数据，数据具体信息在配置文件PRI_signal.ini中配置。通过修改rayleigh_signal变量和bkp_points变量来修改数据段的结构。

PRI_stagger：生成多功能雷达参差抖动PRI数据，数据具体信息在配置文件PRI_signal.ini中配置。通过修改stagger_signal_disorder，stagger_signal_isorder变量和bkp_points变量来修改数据段的结构。其中参数isorder=True表示PRI参差抖动PRI脉冲有序，isorder=False表示无序。

add_spur_PRI：给时间序列添加虚假脉冲，可以通过脉冲密度或者脉冲比例来添加虚假脉冲。

miss_PRI：在脉冲中随机剔除脉冲，构造缺失脉冲。

del_outlier：检测PRI雷达脉冲中的缺失脉冲和虚假脉冲，并删除脉冲。

### 2）how to use：
在PRI_signal.ini中修改PRI脉冲信号的参数
在dataset.py中修改以下部分。
```
if __name__ == '__main__':

    #生成高斯抖动PRI脉冲 
    dataset = datasets()
    signal, bkps, mean, var = dataset.PRI_Gauss_Jitter()

    # #在高斯抖动PRI脉冲中加入虚假脉冲
    PRI_spur = dataset.add_spur_PRI(signal, para=0.001, mode='pulse_ratio')    #添加虚假脉冲

    #在高斯抖动PRI脉冲中删除一些脉冲
    # PRI_miss = dataset.miss_PRI(signal, miss_ratio=0.05)

    #绘图
    plt.scatter(range(len(PRI_spur)), PRI_spur, marker='+', color='b')
    plt.ylim(20, 125)
    plt.xlabel("time/s")
    plt.ylabel("amplitude")
    plt.show()
```
终端运行下列命令
```
python dataset.py
```

### 3）TODO：
使用parser

## 切换点检测算法ChangeFinder
### 1）description
class ChangeFinder:采用SDAR算法对序列进行拟合，并通过ChangeFinder框架检测离群点和切换点。
为实时检测算法，每次读取一个数据，计算出打分。
### 2) how to use
修改ChangeFinder的主函数部分。
```
if __name__ == '__main__':
    cf = changefinder.ChangeFinder(r=0.01, order=1, smooth=7, outlier=False)

    ret = []
    for i in data:
        score = cf.update(i)
        ret.append(score)

    display.display_signal_score(signal_pure, ret, mode='PRI')
```
在终端运行下列命令
```
python ChangeFinder.py
```
### 3) TODO:
使用parser

## 切换点检测算法ChangeFinder_ARIMA
### 1）description
class ChangeFinder_ARIMA:采用ARIMA算法对序列进行拟合，并通过ChangeFinder框架检测离群点和切换点,由于加入了差分步骤，差分数据平稳，平稳数据检测效果提升。
为实时检测算法，每次读取一个数据，计算出打分。
### 2) how to use
修改ChangeFinder的主函数部分。
```
if __name__ == '__main__':
    cf = changefinder.ChangeFinder_ARIMA(term=30, smooth=7, order=(1,0,0), outlier=False)

    ret = []
    for i in data:
        score = cf.update(i)
        ret.append(score)

    display.display_signal_score(signal_pure, ret, mode='PRI')
```
在终端运行下列命令
```
python ChangeFinder.py
```
### 3) TODO:
使用parser

## 切换点检测算法ChangeFinder_ARIMA
### 1）description
class ChangeFinder_ARIMA:采用ARIMA算法对序列进行拟合，并通过ChangeFinder框架检测离群点和切换点,由于加入了差分步骤，差分数据平稳，平稳数据检测效果提升。
为实时检测算法，每次读取一个数据，计算出打分。
### 2) how to use
修改ChangeFinder的主函数部分。
```
if __name__ == '__main__':
    cf = changefinder.ChangeFinder_ARIMA(term=30, smooth=7, order=(1,0,0), outlier=False)

    ret = []
    for i in data:
        score = cf.update(i)
        ret.append(score)

    display.display_signal_score(signal_pure, ret, mode='PRI')
```
在终端运行下列命令
```
python ChangeFinder.py
```
### 3) TODO:
使用parser

## 切换点检测算法FSS
### 1）description
class FSS:预先设定fixed_size 和 threshold，这两个参数设置时应该成正比，具体实验记录见实验报告。
### 2) how to use
修改ChangeFinder的主函数部分。在参数中预先设置是否已知切换点前后参数 通过修改para_known实现。
```
#FSS
detector = FSS(signal_pure, bkps=[], mean=[], var=[], para_known=False, fixed_threshold=3000, fixed_size=10)
indicater = detector.fss_detection()
```
在终端运行下列命令
```
python ChangeFinder.py
```
### 3) TODO:
使用parser

## 切换点检测算法CUSUM
### 1）description
class CUSUM:预先设定threshold，这两个参数设置时应该成正比，具体实验记录见实验报告。
### 2) how to use
修改ChangeFinder的主函数部分。在参数中预先设置是否已知切换点前后参数 通过修改para_known实现。
```
detector = CUsum(bkps, mean, var)
for sig in signal:
    skt, bkp_detect = detector.update(sig)
```
在终端运行下列命令
```
python ChangeFinder.py
```
### 3) TODO:
使用parser

## 对于雷达数据首先删除虚假脉冲和缺失脉冲，再进行切换点检测 main.py
### 1）description
通过ChangeFinder框架的第一部分进行虚假脉冲和缺失脉冲检测，之后进行切换点检测。
### 2）how to use
在main.py中挑选使用相应的指令完成不同算法的应用与不同数据集合的使用。
```
import numpy as np

import ChangeFinder as CF 
from ChangeFinder import CUsum
from ChangeFinder import FSS
import display
import dataset
import warnings
warnings.filterwarnings('ignore')
##################################################################################################
####################################### 生成数据集 #################################################
##################################################################################################
dataset = dataset.datasets(1000, 10)

# 生成均值跳跃数据集
# signal, bkps, outliers = dataset.jumping_mean_random(n_samples=1000, n_bkps=10)
# 生成均值方差跳跃数据集
# signal, bkps = dataset.jumping_mean_variance(n_samples=1000, n_bkp=10)
# 生成高斯抖动PRI脉冲
signal, bkps, mean, var = dataset.PRI_Gauss_Jitter()


# 高斯抖动PRI脉冲中添加缺失脉冲
# PRI_miss = dataset.miss_PRI(signal, miss_ratio=0.01)
# 高斯抖动PRI脉冲中添加虚假脉冲
PRI_spur = dataset.add_spur_PRI(signal, para=0.0002, mode='pulse_ratio')    #添加虚假脉冲

# 从时间序列中检测离群点
# cf = CF.ChangeFinder(r=0.007, order=1, smooth=5, outlier=True)
# 生成检测器
cf = CF.ChangeFinder_ARIMA(term=20, smooth=5, order=(0, 0, 1), outlier=True)
# cf = CF.ChangeFinder(r=0.008, order=1, smooth=7, outlier=False)

ret = []
points = []
estimate_bkp = []

# 发现离群点
for index, data in enumerate(PRI_spur):
    score = cf.update(data)
    # if score[0] < 0:
    #     ret.append(0)
    #     continue
    if score[0] > 20:
        points.append(index)
    ret.append(score[0])

# 删除离群点
# signal_pure = dataset.del_outlier(signal, points)
##################################################################################################
####################################### 生成数据集 #################################################
##################################################################################################

##################################################################################################
####################################### 切换点检测 #################################################
##################################################################################################
#cusum
# detector = CUsum(bkps=[], mean=[], var=[], para_known=False)
# score = []
# for sig in signal_pure:
#     scor = detector.update(sig)
#     score.append(scor)

#FSS
# detector = FSS(signal_pure, bkps=[], mean=[], var=[], para_known=False, fixed_threshold=3000, fixed_size=10)
# indicater = detector.fss_detection()

#changefinder
##################################################################################################
####################################### 切换点检测 #################################################
##################################################################################################

##################################################################################################
####################################### 结果显示 ##################################################
##################################################################################################
# 显示离群点检测的结果
display.display_signal_score(PRI_spur, ret, mode='PRI')
# 显示清洗之后的信号
# display.display_signal_clean(PRI_spur, signal_pure)
# CUSUM显示信号以及打分
# display.display_signal_score(signal_pure, score, mode='PRI')
# FSS 显示信号以及打分
# display.display_signal_score(signal_pure, indicater, mode='PRI')
# ChangeFInder显示信号以及打分
# display.display_signal_score(signal_pure, ret, mode='PRI')
# 显示pr曲线
# display.PR_cruve(bkps, ret)
```
在终端命令行中运行以下指令
```
python main.py
```

## Reference
1.非合作观测下的MFR雷达状态自动定义，北京理工大学雷达与对抗技术研究所
2.Nikiforov I V . Two strategies in the problem of change detection and isolation[J]. IEEE Transactions on Information Theory, 1997, 43(2):770-776.
3.Takeuchi J, Yamanishi K. A unifying framework for detecting outliers and change points from time series[J]. IEEE transactions on Knowledge and Data Engineering, 2006, 18(4):482-492.
4.Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.