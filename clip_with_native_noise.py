import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I\\O (e.g. pd.read_csv)
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import librosa.display
import noisereduce
import os
import python_speech_features as psf
# Input data files are available in the read-only "..\\input\\" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
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
short_audio_path = "data\\birdclef-2021\\train_short_audio"
birds_dir = glob.glob(short_audio_path+"\\*")[:15]
meta = pd.read_csv("data\\birdclef-2021\\train_metadata.csv")
good_sample = meta.loc[meta.rating==5,]
good_sample = good_sample.loc[good_sample.secondary_labels=='[]',]
good_sample_files = np.array(good_sample.filename)
if not os.path.isdir("data\\birdclef-2021\\features" ):
    os.mkdir("data\\birdclef-2021\\features")
for dirs in birds_dir:
    ogg_files = glob.glob(dirs+"\\*.ogg")
    count = 0
    bird_name = dirs.split("\\")[-1]
    if not os.path.isdir("data\\birdclef-2021\\features\\"+bird_name):
        os.mkdir("data\\birdclef-2021\\features\\"+bird_name)
    for f in ogg_files:
        file_info = f.split("\\")
        if count >= 3:
            continue
        if file_info[-1] not in good_sample_files:
            continue
        destination_dir = 'data\\birdclef-2021\\features\\' + file_info[-2] + '\\' + file_info[-1].split(".")[0]
        audio, sr = librosa.load(f, sr=32000)
        audio_reduced = noisereduce.reduce_noise(audio_clip=audio, noise_clip=audio[:int(0.5*sr)])
        width = 0.15
        hop = width / 10
        map = split_audio(audio_reduced, w=int(width * sr), h=int(hop * sr), threshold_level=1, tolerence=10, sr=sr)
        # fig, ax = plt.subplots()
        # librosa.display.waveplot(audio_reduced, sr, color='blue', ax=ax)
        # for clip in map:
        #     start = clip[0]
        #     end = clip[1]
        #     librosa.display.waveplot(audio_reduced[start:end], sr, offset=start \\ sr, color="red", ax=ax)
        # plt.xlim([0, len(audio_reduced) \\ sr])
        # img_dir = "\\".join(destination_dir.split("\\")[:-1])
        # if not os.path.isdir(img_dir):
        #     os.makedirs(img_dir)
        # fig.savefig(destination_dir+"_wave.jpg")
        # plt.close()
        # wh_list = glob.glob('data\\wh_bg\\*.ogg')
        clip_id = 0
        outside_map = [0] + [y for x in map for y in x] + [len(audio_reduced)]
        outside_map = np.array(outside_map)
        outside_map = outside_map.reshape([int(len(outside_map)/2), 2])
        outside_audio = [audio[x[0]:x[1]] for x in outside_map]
        tmp = []
        for i in outside_audio:
            tmp += list(i)
        outside_audio = tmp
        if len(outside_audio) - 5 * sr <= 0:
            continue
        for clip in map:
            start = clip[0]
            end = clip[1]
            wave_clip = audio[clip[0]:clip[1]]
            clip_sec = len(wave_clip) / sr
            if clip_sec >= 5:
                clip_5s = wave_clip[int((len(wave_clip)-5*sr)/2):int((len(wave_clip)+5*sr)/2)]
            else:
                # extract_wh = np.random.randint(0, len(wh_list) - 1)
                # extract_wh, sr = librosa.load(wh_list[extract_wh], sr=sr)
                # extract_wh = extract_wh * np.mean(np.abs(audio[:int(0.5 * sr)])) \\ np.mean(np.abs(extract_wh))
                # mid = len(extract_wh) \\ 2
                # clip_5s = extract_wh
                # clip_5s[int((len(extract_wh) - len(wave_clip)) \\ 2):int((len(extract_wh) + len(wave_clip)) \\ 2)] = wave_clip
                clip_5s = np.repeat(0,sr*5)
                clip_5s[int((sr*5-len(wave_clip))/2):int((sr*5+len(wave_clip))/2)] = wave_clip
                wn_5s = np.random.randint(0,len(outside_audio)-5*sr)
                clip_5s = np.array([sum(x) for x in zip(clip_5s, outside_audio[wn_5s:(wn_5s+5*sr)])])
            sf.write(destination_dir + "_" + str(clip_id) + ".ogg", clip_5s, samplerate=sr, subtype="VORBIS")
            # win_len = 512
            # hop_len = 256
            # spec = librosa.stft(clip_5s, n_fft=win_len, hop_length=hop_len)
            # spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
            # fig, ax = plt.subplots()
            # librosa.display.specshow(spec_db,
            #                          sr=sr,
            #                          hop_length=hop_len,
            #                          x_axis='time',
            #                          y_axis='hz',
            #                          cmap=plt.get_cmap('viridis'))
            # fig.savefig(destination_dir + "_"+str(clip_id)+".jpg")
            # plt.close()
            # np.save(destination_dir+"_"+str(clip_id)+".npy", spec_db)
            count += 1
            clip_id+=1
