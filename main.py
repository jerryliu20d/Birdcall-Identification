import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import librosa.display
import noisereduce
import os
import python_speech_features as psf
import statistics
import scipy.ndimage as ndimg
import time
#%%
class CFG:
    test_mode = True
#%%
def un_stft(map, end, hop_len):
    if not len(map):
        print('eeeee')
    return_map = [[x[0] * hop_len, (x[1] + 1) * hop_len] for x in map]
    if map[-1][-1] == end:
        return_map[-1][-1] = end
    return return_map


def map_func(dia_indicator):
    map=[]
    flag = False
    for i, v in enumerate(dia_indicator):
        if v:
            if not flag:
                start = i
                flag = True
        else:
            if flag:
                end = i-1
                flag = False
                map.append([start, end])
    if flag:
        map.append([start, i])
    return map


def spec_morph(wav, sr=32000, n_fft=512, hop_len=int(512*.75), median_amplify=3, open_structure=np.ones((15, 5)), dia_structure=np.ones(50), signal=True,
               plot_flag=False):
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_len, window='hann')
    stft = np.abs(stft)
    stft = stft/np.max(stft)
    rowwise = np.array([x > median_amplify*statistics.median(x) for x in stft])
    t_stft = stft.transpose()
    colwise = np.array([x > median_amplify*statistics.median(x) for x in t_stft]).transpose()
    select_pixels = np.logical_and(rowwise, colwise)
    # ero_pix = ndimg.binary_erosion(select_pixels, structure=np.array([[0,0,1,0,0],
    #                                                                   [0,1,1,1,0],
    #                                                                   [1,1,1,1,1],
    #                                                                   [0,1,1,1,0],
    #                                                                   [0,0,1,0,0]],dtype=int))
    ero_pix = ndimg.binary_erosion(select_pixels, structure=open_structure)
    dia_pix = ndimg.binary_dilation(ero_pix, structure=open_structure)
    indicator = dia_pix.transpose()
    indicator = np.array([np.any(x) for x in indicator])
    dia_indicator = ndimg.binary_dilation(indicator, structure=dia_structure, iterations=2)
    if not signal:
        dia_indicator = np.logical_not(dia_indicator)
    map = map_func(dia_indicator)
    sr_map = un_stft(map, len(dia_indicator), hop_len)
    if plot_flag == True:
        librosa.display.specshow(dia_pix,
                                 sr=sr,
                                 hop_length=hop_len,
                                 x_axis='time',
                                 y_axis='hz',
                                 cmap=plt.get_cmap('gray_r'))
        plt.show()
    return sr_map, stft
#%%
# audio, sr = librosa.load('data/birdclef-2021/train_short_audio/acafly/XC6671.ogg', sr=32000)
# audio, sr = librosa.load('data/birdclef-2021/train_short_audio/acowoo/XC448737.ogg', sr=32000)
start = time.time()
audio, sr = librosa.load('data/birdclef-2021/train_soundscapes/51010_SSW_20170513.ogg', sr=32000)
audio = audio[int(4.5*60*sr+5*sr*5):int(4.5*60*sr+5*sr*6)]
dia_structure = np.ones(8)
open_structure = np.ones((2, 2))
# open_structure = np.array([[0,0,1,0,0]]*2+[[0,1,1,1,0]]*2+[[1,1,1,1,1]]*5+[[0,1,1,1,0]]*2+[[0,0,1,0,0]]*2)
signal_cut, stft = spec_morph(wav=audio, signal=True, median_amplify=3, dia_structure=dia_structure, open_structure=open_structure, plot_flag=False)
noise_cut, stft = spec_morph(wav=audio, median_amplify=2.5, signal=False, dia_structure=dia_structure)
end=time.time()
print(end-start)
fig, ax1 = plt.subplots()
librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft)),
                         sr=sr,
                         hop_length=int(0.75 * 512),
                         x_axis='time',
                         y_axis='hz',
                         cmap=plt.get_cmap('gray_r'))
ax2=ax1.twinx()
for xlim in signal_cut:
    ax2.hlines(y=1, xmin=xlim[1]/sr, xmax=xlim[0]/sr, colors='red')
for xlim in noise_cut:
    ax2.hlines(y=0, xmin=xlim[1]/sr, xmax=xlim[0]/sr, colors='red')
plt.show()


fig, ax1 = plt.subplots()
librosa.display.waveplot(audio, sr=sr)
ax2=ax1.twinx()
for xlim in signal_cut:
    ax2.hlines(y=1, xmin=xlim[1]/sr, xmax=xlim[0]/sr, colors='red')
for xlim in noise_cut:
    ax2.hlines(y=0, xmin=xlim[1]/sr, xmax=xlim[0]/sr, colors='red')
plt.show()


