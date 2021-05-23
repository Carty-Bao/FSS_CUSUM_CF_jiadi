import numpy as np

import ChangeFinder as CF 
from ChangeFinder import CUsum
from ChangeFinder import FSS
import display
import dataset
import warnings
import time
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
# signal, bkps, mean, var = dataset.PRI_Gauss_Jitter()
# 生成均匀抖动PRI脉冲
signal, bkps, lower_bound, upper_bound = dataset.PRI_norm_Jitter()
# 生成锐利抖动PRI脉冲
# signal, bkps, scale = dataset.PRI_rayleigh_Jitter()

# 高斯抖动PRI脉冲中添加缺失脉冲
# PRI_miss = dataset.miss_PRI(signal, miss_ratio=0)
# 高斯抖动PRI脉冲中添加虚假脉冲
# PRI_spur = dataset.add_spur_PRI(signal, para=0.0002, mode='pulse_ratio')    #添加虚假脉冲

# 从时间序列中检测离群点
# cf = CF.ChangeFinder(r=0.007, order=1, smooth=5, outlier=True)
# 生成检测器
# cf = CF.ChangeFinder_ARIMA(term=20, smooth=5, order=(0, 0, 1), outlier=True)
# cf = CF.ChangeFinder(r=0.008, order=1, smooth=7, outlier=False)

ret = []
points = []
estimate_bkp = []

# 发现离群点
# for index, data in enumerate(PRI_spur):
#     score = cf.update(data)
#     # if score[0] < 0:
#     #     ret.append(0)
#     #     continue
#     if score[0] > 20:
#         points.append(index)
#     ret.append(score[0])

# 删除离群点
# signal_pure = dataset.del_outlier(signal, points)
##################################################################################################
####################################### 生成数据集 #################################################
##################################################################################################

##################################################################################################
####################################### 切换点检测 #################################################
##################################################################################################
# cusum
# detector = CUsum(bkps=bkps, mean=mean, var=var, para_known=False)
# score = []
# start = time.time()
# for sig in PRI_miss:
#     scor = detector.update(sig)
#     score.append(scor)
# end = time.time()

#FSS
# detector = FSS(signal, bkps=bkps, mean=mean, var=var, para_known=False, fixed_threshold=80, fixed_size=7)
# start = time.time()
# indicater = detector.fss_detection()
# end = time.time()

#changefinder
detector = CF.ChangeFinder(r=0.008, order=1, smooth=5, outlier=False)
for index, data in enumerate(signal):
    score = detector.update(data)
    # if score[0] < 0:
    #     ret.append(0)
    #     continue
    # if score[0] > 20:
    #     points.append(index)
    ret.append(score[0])
##################################################################################################
####################################### 切换点检测 #################################################
##################################################################################################

##################################################################################################
####################################### 结果显示 ##################################################
##################################################################################################
# 显示离群点检测的结果
# display.display_signal_score(PRI_spur, ret, mode='PRI')

# 显示清洗之后的信号
# display.display_signal_clean(PRI_spur, signal_pure)

# CUSUM显示信号以及打分
# display.display_signal_score(PRI_miss, score, mode='PRI')

# FSS 显示信号以及打分
# display.display_signal_score(PRI_miss, indicater, mode='PRI')

# ChangeFInder显示信号以及打分
display.display_signal_score(signal, ret, mode='PRI')

# 显示pr曲线
# display.PR_cruve(bkps, ret)