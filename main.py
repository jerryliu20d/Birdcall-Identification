import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import librosa.display
import noisereduce
import os
#%%
def split_audio(audio_data, w, h, threshold_level, tolerence=10, sr=32000):
    split_map = []
    start = 0
    data = np.abs(audio_data)
    threshold = threshold_level*(np.mean(data[:int(.5*sr)])+np.mean(data))*0.5
    inside_sound = False
    near = 0
    for i in range(0, len(data)-w, h):
        win_mean = np.mean(data[i:i+w])
        if(win_mean>threshold and not(inside_sound)):
            inside_sound = True
            start = i
        if(win_mean<=threshold and inside_sound and near>tolerence):
            inside_sound = False
            near = 0
            split_map.append([max(start-near, 0), i])
        if(inside_sound and win_mean<=threshold):
            near += 1
    return split_map
#%%
bg_table = pd.read_csv('data/train_soundscape_labels.csv')
file_names = glob.glob('data\\train_soundscapes\\*.ogg')
for f in file_names:
    audio, sr = librosa.load(f, sr=32000)
    audio_id = int(f.split("\\")[2].split('_')[0])
    target_list = bg_table.loc[bg_table.audio_id==audio_id,]
    target_list = target_list.loc[target_list.birds=='nocall',]
    for time in target_list.seconds:
        clip = audio[(time-5)*sr:time*sr]
        sf.write(data=clip, file="data\\wh_bg\\"+str(audio_id)+"_"+str(time)+".ogg", samplerate=sr, format='ogg', subtype='vorbis')

#%%
short_audio_path = "data\\train_short_audio"
birds_dir = glob.glob(short_audio_path+"\\*")
for dir in birds_dir:
    ogg_files = glob.glob(dir+"\\*.ogg")
    for f in ogg_files:
        file_info = f.split("\\")
        destination_dir = 'data\\features\\' + file_info[2] + '\\' + file_info[3].split(".")[0]
        audio, sr = librosa.load(f, sr=32000)
        audio_reduced = noisereduce.reduce_noise(audio_clip=audio, noise_clip=audio[:int(0.5*sr)])
        width = 0.15
        hop = width / 10
        map = split_audio(audio_reduced, w=int(width * sr), h=int(hop * sr), threshold_level=1, tolerence=10, sr=sr)
        fig, ax = plt.subplots()
        librosa.display.waveplot(audio_reduced, sr, color='blue', ax=ax)
        for clip in map:
            start = clip[0]
            end = clip[1]
            librosa.display.waveplot(audio_reduced[start:end], sr, offset=start / sr, color="red", ax=ax)
        plt.xlim([0, len(audio_reduced) / sr])
        img_dir = "\\".join(destination_dir.split("\\")[:-1])
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        fig.savefig(destination_dir+"_wave.jpg")
        wh_list = glob.glob('data\\wh_bg\\*.ogg')
        clip_id = 0
        for clip in map:
            start = clip[0]
            end = clip[1]
            wave_clip = audio[clip[0]:clip[1]]
            clip_sec = len(wave_clip) / sr
            if clip_sec >= 5:
                clip_5s = wave_clip[int((len(wave_clip)-5*sr)/2):int((len(wave_clip)+5*sr)/2)]
            else:
                extract_wh = np.random.randint(0, len(wh_list) - 1)
                extract_wh, sr = librosa.load(wh_list[extract_wh], sr=sr)
                extract_wh = extract_wh * np.mean(np.abs(audio[:int(0.5 * sr)])) / np.mean(np.abs(extract_wh))
                mid = len(extract_wh) / 2
                clip_5s = extract_wh
                clip_5s[int((len(extract_wh) - len(wave_clip)) / 2):int((len(extract_wh) + len(wave_clip)) / 2)] = wave_clip
            win_len = 512
            hop_len = 256
            spec = librosa.stft(clip_5s, n_fft=win_len, hop_length=hop_len)
            spec_db = librosa.amplitude_to_db(spec, ref=np.max)
            fig, ax = plt.subplots()
            librosa.display.specshow(spec_db, 
                                     sr=sr,
                                     hop_length=hop_len,
                                     x_axis='time',
                                     y_axis='hz',
                                     cmap=plt.get_cmap('viridis'))
            fig.savefig(destination_dir + "_"+str(clip_id)+".jpg")
            np.save(destination_dir+"_"+str(clip_id)+".npy", spec_db)
            clip_id+=1
