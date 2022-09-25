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

from fcmeans import FCM   # pip install fuzzy-c-means
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import os
import csv


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

    
    
def sample_entropy(time_series, sample_length, tolerance=None):
    """Calculates the sample entropy of degree m of a time_series.
    This method uses chebychev norm.
    It is quite fast for random data, but can be slower is there is
    structure in the input time series.
    Args:
    time_series: numpy array of time series
    sample_length: length of longest template vector
    tolerance: tolerance (defaults to 0.1 * std(time_series)))
    Returns:
    Array of sample entropies:
      SE[k] is ratio "#templates of length k+1" / "#templates of length k"
      where #templates of length 0" = n*(n - 1) / 2, by definition
    Note:
    The parameter 'sample_length' is equal to m + 1 in Ref[1].
    References:
    [1] http://en.wikipedia.org/wiki/Sample_Entropy
    [2] http://physionet.incor.usp.br/physiotools/sampen/
    [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
      of biological signals
    """
    # The code below follows the sample length convention of Ref [1] so:
    M = sample_length - 1

    time_series = np.array(time_series)
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)

    # Ntemp is a vector that holds the number of matches. N[k] holds matches templates of length k
    Ntemp = np.zeros(M + 2)
    # Templates of length 0 matches by definition:
    Ntemp[0] = n * (n - 1) / 2

    for i in range(n - M - 1):
        template = time_series[i:(i + M + 1)]  # We have 'M+1' elements in the template
        rem_time_series = time_series[i + 1:]

        search_list = np.arange(len(rem_time_series) - M, dtype=np.int32)
        for length in range(1, len(template) + 1):
            hit_list = np.abs(rem_time_series[search_list] - template[length - 1]) < tolerance
            Ntemp[length] += np.sum(hit_list)
            search_list = search_list[hit_list] + 1

    sampen = -np.log(Ntemp[1:] / Ntemp[:-1])
    return sampen

def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series
    Args:
    time_series: Time series
    scale: Scale factor
    Returns:
    Vector of coarse-grained time series with given scale factor
    """
    time_series = time_series[0]
    n = len(time_series)
    b = int(np.fix(n/scale))
    temp = np.reshape(time_series[0:b * scale], (b,scale))
    cts = np.mean(temp, axis=1)
    return cts

def multiscale_entropy(time_series, sample_length, tolerance=None, maxscale=None):
    """Calculate the Multiscale Entropy of the given time series considering
    different time-scales of the time series.
    Args:
    time_series: Time series for analysis
    sample_length: Bandwidth or group of points
    tolerance: Tolerance (default = 0.1*std(time_series))
    Returns:
    Vector containing Multiscale Entropy
    """

    if tolerance is None:
        # We need to fix the tolerance at this level
        # If it remains 'None' it will be changed in call to sample_entropy()
        tolerance = 0.1 * np.std(time_series)
    if maxscale is None:
        maxscale = len(time_series)
    mse = np.zeros(maxscale)
    for i in range(maxscale):
        temp = util_granulate_time_series(time_series, i + 1)
        mse[i] = sample_entropy(temp, sample_length, tolerance)[-1]
    return mse   
    
    
    
    
    
def get_features(audio,sr):
    feature_set = []
    # mfcc, zcr, 
    # mfcc刻画音频静态特征
    # mfcc的第一个分量与音频的振幅有关，因此为了排除振幅的干扰，可直接舍弃第一个分量
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, hop_length=1024, n_mfcc=30)  # 提取mfcc特征个数为n_mfcc个 [n_mfcc, frames]
    mfccs_mean = np.mean(mfccs[1:,:], axis=1)
    # 平均后mfccs的维度为[n_mfcc-1,1]
    # mfcc的一阶和二阶导数，刻画音频动态特征
    mfcc_delta = np.mean(librosa.feature.delta(mfccs),axis=1)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfccs,order=2),axis=1)
    zcr =librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=1024)  # 维度随声音事件长度而变化
    mean_zcr = np.mean(zcr)
    
    # multiscale entropy of zcr
    mse_val = multiscale_entropy(zcr, sample_length=3, maxscale=20) 
    
    # 统计特征
    # 偏度与峰度，计算波形与正态分布的相似性
    # 考虑到鼾声波形的正半部分分布较为对称，因此可对其正数部分计算偏度和峰度
    positive_audio = audio[audio>0]
    skew = stats.skew(positive_audio)  # 偏度
    kurtosis = stats.kurtosis(positive_audio)  # 峰度
    
    # 考虑到鼾声波形中间部分高，两边低，因此可计算中间部分能量占总信号的能量比
    # 由于此时一阶与二阶范数等价，因此此处使用一阶范数
    middle_energy_ratio = np.linalg.norm(audio[len(audio)//3:-len(audio)//3], ord = 1)/np.linalg.norm(audio, ord = 1)
    
    feature_set.append(mfccs_mean)
    feature_set.append(mfcc_delta)
    feature_set.append(mfcc_delta2)
    feature_set.append(mse_val)
    feature_set.append(skew)
    feature_set.append(kurtosis)
    feature_set.append(mean_zcr)
    
    
    
    # 检查特征绝对值，决定是否进行特征归一化处理
    return feature_set
    
    

if __name__ == '__main__':

    
    audio_dir ="/mnt/disk1/sjp/SPCdevkit/SPC2018/soundEvent6cls"

    labels = {'snore':1, 'speech':2, 'cough':3, 'knock':4, 'insp':5, 'exp':6}
    filename_list = []
    label_list = []
    feature_list = []
    for root, dirname, filenames in os.walk(audio_dir):
        for filename in filenames:
            label = filename.split('-',1)[0]
            filename_list.append(filename)
            label_list.append(labels[label])
            audio_file = os.path.join(audio_dir,filename)
            audio, sr = librosa.load(audio_file,sr=None)
            feature = get_features(audio, sr)
            feature_list.append(feature)
    
    
    reader = Audio('../data/snore/jingpeng', '141')
    feat = reader.get_feat_df(0.1, 0.05)
    
    clips = sed(audio)  # 其中包含了检测得到的若干个长度不一的声音片段
    
    clips_feature = []
    for clip in clips:
        feat = get_features(clip)
        clips_feature.append(feat)
    
    X = feature_set
    # PCA 降维
    pca = PCA(n_components=len(feature_set))  # 先计算一次特征值
    pca.fit(X)
    # 输出特征值
    print(pca.explained_variance_)
    fig = plt.figure()
    plt.plot(pca.explained_variance_)
    plt.show()
    # 输出特征向量
#     print(pca.components_)
    
    decreased_dim = 3
    pca = PCA(n_components=decreased_dim)  # 降到decreased_dim 维
    pca.fit(X)
    
    # 降维后的数据
    X_new = pca.transform(X)
#     print(X_new)
    # 画图显示数据
    if decreased_dim == 2:
        fig = plt.figure()
        plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
        plt.show()
    elif decreased_dim == 3:
        # 三维画图
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
        plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
        plt.show()
        
    # 聚类实现鼾声检测
    # fitting the fuzzy-c-means
    fcm = FCM(n_clusters=2)
    fcm.fit(X_new)
    
    # outputs 
    fcm_centers = fcm.centers
    fcm_labels = fcm.predict(X_new)
    
    
    # plot results
    f, axes = plt.subplots(1, 2, figsize=(11,5))
    axes[0].scatter(X_new[:,0], X_new[:,1], alpha=.1)
    axes[1].scatter(X_new[:,0], X_new[:,1], c=fcm_labels, alpha=.1)
    axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
    plt.savefig('clustering-output.jpg')
    plt.show()
    
    
    
    
    
    
    
    
    
    headers = ['uid', 'label', 
               'mfcc-2-mean','mfcc-3-mean','mfcc-4-mean','mfcc-5-mean','mfcc-6-mean','mfcc-7-mean','mfcc-8-mean','mfcc-9-mean',
               'mfcc-10-mean','mfcc-11-mean','mfcc-12-mean','mfcc-13-mean','mfcc-14-mean','mfcc-15-mean','mfcc-16-mean', 'mfcc-17-mean',
               'mfcc-18-mean','mfcc-19-mean','mfcc-20-mean',
               'mfcc-delta-1-mean','mfcc-delta-2-mean','mfcc-delta-3-mean','mfcc-delta-4-mean','mfcc-delta-5-mean','mfcc-delta-6-mean',
               'mfcc-delta-7-mean','mfcc-delta-8-mean','mfcc-delta-9-mean','mfcc-delta-10-mean','mfcc-delta-11-mean','mfcc-delta-12-mean',
               'mfcc-delta-13-mean','mfcc-delta-14-mean','mfcc-delta-15-mean','mfcc-delta-16-mean','mfcc-delta-17-mean','mfcc-delta-18-mean',
               'mfcc-delta-19-mean','mfcc-delta-20-mean',
               'mfcc-delta2-1-mean','mfcc-delta2-2-mean','mfcc-delta2-3-mean','mfcc-delta2-4-mean','mfcc-delta2-5-mean','mfcc-delta2-6-mean',
               'mfcc-delta2-7-mean','mfcc-delta2-8-mean','mfcc-delta2-9-mean','mfcc-delta2-10-mean','mfcc-delta2-11-mean','mfcc-delta2-12-mean',
               'mfcc-delta2-13-mean','mfcc-delta2-14-mean','mfcc-delta2-15-mean','mfcc-delta2-16-mean','mfcc-delta2-17-mean','mfcc-delta2-18-mean',
               'mfcc-delta2-19-mean','mfcc-delta2-20-mean',
               'mse-1','mse-2','mse-3','mse-4','mse-5','mse-6','mse-7','mse-8','mse-9','mse-10',
               'mse-11','mse-12','mse-13','mse-14','mse-15','mse-16','mse-17','mse-18','mse-19','mse-20',
               'skew','kurtosis','mean-zcr']
    uid = filename_list
    label = label_list
    rows = np.hstack((uid.T,label.T,feature_set))
    
    file = open("features.csv","w",encoding="utf-8",newline='')
    write = csv.writer(file)
    write.writerow(headers)
    write.writerows(rows)
    file.close()
    
    
    
    
    print()
