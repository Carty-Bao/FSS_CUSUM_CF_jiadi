import os

from numpy import random as rd 
import numpy as np

from dataset import datasets as datasets
import display

from ChangeFinder import CUsum    
from ChangeFinder import FSS



if __name__ == '__main__':
    data = datasets()
    signal, bkps, mean, var = data.PRI_Gauss_Jitter()

    # display.display_signal_score(signal)
    # detector = FSS(signal, bkps=[], mean=[], var=[], para_known=False, fixed_threshold=1000, fixed_size=50)
    # indicater = detector.fss_detection()
    detector = CUsum(bkps=[], mean=[], var=[], para_known=False)
    score = []
    for sig in signal:
        scor = detector.update(sig)
        score.append(scor)

    display.display_signal_score(signal, score, mode='PRI')





        
    