#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   features  
@Time        :   2022/9/22 14:18
@Author      :   Jingpeng Sun
@Description :   
"""
import librosa
import xml.etree.ElementTree as ET

from scipy.signal import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from librosa.feature import zero_crossing_rate
from utils import energy
import matplotlib; matplotlib.use('TkAgg')

class Audio:
    def __init__(self, path, uid, sfreq=16000):
        self.wav, self.sfreq = librosa.load(f'{path}/{uid}.wav', sr=sfreq)
        self.wav = savgol_filter(self.wav, int(sfreq/100 + 1), 5)
        self.num_of_samps = len(self.wav)
        self.anno = ET.ElementTree(file=f'{path}/{uid}.xml')

    def get_feat_df(self, window, step, feats=None):
        '''
        :param window: in seconds
        :param step: in seconds
        :param feats:
        :return:
        '''

        window = int(window * self.sfreq)   # in num of samps
        step = int(step * self.sfreq)       # in num of samps
        if feats is None:
            feats = ['energy', 'zcr']

        index = [idx / self.sfreq for idx in range(0, self.num_of_samps-window+1, step)]

        feat_list = []
        for feat in feats:
            if feat  == 'energy':
                _energy = energy(self.wav, frame_length=window, hop_length=step)
                feat_list.append(_energy)
            elif feat == 'zcr':
                _zcr = zero_crossing_rate(self.wav, frame_length=window, hop_length=step, center=False)
                feat_list.append(np.squeeze(_zcr))

        res = pd.DataFrame(
            np.array(feat_list).T, index=index, columns=feats
        )

        return res

if __name__ == '__main__':

    reader = Audio('../data/snore/jingpeng', '141')
    feat = reader.get_feat_df(0.1, 0.05)
    plt.subplot(211)
    plt.plot(reader.wav)
    plt.subplot(212)
    feat['energy'] = savgol_filter(feat['energy'], 11, 9)
    plt.plot(feat['energy'])
    plt.plot(feat['zcr'])
    plt.show()
    print()
