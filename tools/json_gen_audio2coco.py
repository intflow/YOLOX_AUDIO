#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Intflow, Inc. and its affiliates.

# %%
import os
import json
import tqdm
import numpy as np
import cv2
import sys
c_folder = os.path.abspath(os.path.dirname(__file__))
p_folder = os.path.abspath(os.path.dirname(c_folder))
pp_folder = os.path.abspath(os.path.dirname(p_folder))
sys.path.append(c_folder)
sys.path.append(p_folder)
sys.path.append(pp_folder)
import yolox.utils.boxes as B
import json
from scipy.io import wavfile
import scipy.io
import librosa
from PIL import Image

root = '/data/AIGC_3rd_2021/GIST_tr2_veryhard500'
os.system('rm -rf '+root+'/img/')
os.system('mkdir '+root+'/img/')
wav_folder_path = os.path.join(root, 'wav')
img_folder_path = os.path.join(root, 'img')
train_label_path = os.path.join(root, 'tr2_devel_500.json')
train_label_merge_out = os.path.join(root, 'label_coco_bbox.json')

mode = 1   #  0:train_data,   1:validation_data

with open(train_label_path, 'r') as j:
     contents = json.loads(j.read())

json_list = list(contents)

# %%
merged_label = {}
images = []
annotations = []

cat_dict = {
    "M" : 0,
    "W" : 1,
    "C" : 2
}


def _nz(x):
    if x < 0.0:
        x = 0.0
    return x

def annot_overlay_stft(img, dets):
        category_dic={0:'M',1:'W',2:'C'} #class name
        category_color={0:(255,0,0),1:(0,255,0),2:(0,0,255)} #class color

        for det in dets:
            
            x1=det[0]    #x
            x2=det[1]    #y
            width=det[2]   #width
            height=det[3]  #height

            category_id=int(det[4])
            try:
                cv2.rectangle(img, (x1, y1), (x1+width, y1+height), category_color[category_id], 2)
                cv2.putText(img, category_dic[category_id], (int(x1-10),int(y1-10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, category_color[category_id], 1)
            except:
                print('[False Dataset!] ', x1, y1, width, height)
           
        return img

img_id = 0
obj_id = 0
for num1, each_file in enumerate(tqdm.tqdm(json_list)):
    
    wav_label = contents[each_file]
    
    # Load wav file
    wav_path = os.path.join(wav_folder_path, each_file)
    try:
        sr, wavs = wavfile.read(wav_path)
    except:
        print('Can not load wav file')
        continue

    # 7 to 1 sum > Resample to 8K > Get spectrogram > chop to multiple images
    step = int(sr / 6000)
    wav = np.mean(wavs,1)
    wav_6k = wav[::step] / 32768.0
    sr = 6000

    # STFT -> spectrogram
    hop_length = 128  # 전체 frame 수 (21.33ms)
    n_fft = 512  # frame 하나당 sample 수 (85.33ms)
    hop_length_s = hop_length / sr

    # calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length)/sr
    n_fft_duration = float(n_fft)/sr

    # STFT
    stft = librosa.stft(wav_6k, n_fft=n_fft, hop_length=hop_length)
    stft = stft[1:,:]
    f_len = stft.shape[0]
    t_len = stft.shape[-1]
    t_step = 512
    c_len = 3

    # 복소공간 값 절댓값 취하기
    mel = librosa.feature.melspectrogram(S=np.abs(stft), sr=sr, n_mels=f_len).reshape(f_len, -1, 1)
    mag = np.abs(stft).reshape(f_len, -1, 1)
    mfcc = librosa.feature.mfcc(S=mel, sr=sr, n_mfcc=f_len)

    feat_cat = np.concatenate((mel, mag, mfcc), axis = 2)
    feat_cat = feat_cat ** 2.0

    # magnitude > Decibels
    for i in range(0,3):
        feat = feat_cat[:,:,i]
        log_spectrogram = librosa.amplitude_to_db(feat)
        log_spectrogram += np.abs(log_spectrogram.min()) + 1e-5
        log_spectrogram /= log_spectrogram.max()
        log_spectrogram *= 255.0
        log_spectrogram = np.flip(log_spectrogram, axis=0)
        feat_cat[:,:,i] = log_spectrogram

    sub_cnt = 0
    img_set = []
    for t_sub in range(0, t_len, t_step):
        t_end = t_sub + t_step
        
        if t_end > t_len:
            feat_cat_sub = np.concatenate((feat_cat[:,t_sub:,:], np.zeros((f_len, t_end - t_len, c_len))),axis=1)
        else:
            feat_cat_sub = feat_cat[:,t_sub:t_end,:]

        feat_cat_sub = feat_cat_sub.astype('uint8')
        img = Image.fromarray(feat_cat_sub)
        img_path = os.path.join(img_folder_path, each_file)[:-4]+'_'+str(sub_cnt)+'.jpg'
        

        # organize ['result']['objects']
        dets = []

        t_sub_s = t_sub * hop_length_s
        t_end_s = t_end * hop_length_s
        for num2, each_vad in enumerate(wav_label['on_offset']):
            if t_sub_s <= each_vad[0] and t_end_s >= each_vad[1]:
                 x1 = int(each_vad[0] / hop_length_s) % t_step
                 x2 = int(each_vad[1] / hop_length_s) % t_step
            elif t_sub_s <= each_vad[0] and t_end_s > each_vad[0] and t_end_s < each_vad[1]:
                 x1 = int(each_vad[0] / hop_length_s) % t_step
                 x2 = int(t_end-1) % t_step
            elif t_sub_s > each_vad[0] and t_sub_s < each_vad[1] and t_end_s >= each_vad[1]:
                 x1 = int(t_sub) % t_step
                 x2 = int(each_vad[1] / hop_length_s) % t_step
            elif t_sub_s > each_vad[0] and t_end_s < each_vad[1]:
                 x1 = int(t_sub) % t_step
                 x2 = int(t_end-1) % t_step
            else:
                continue


            category_id = cat_dict[wav_label['speaker'][num2]]

            y1 = 0
            y2 = 256
            width = x2 - x1
            height = y2 - y1

            if width < 10: # if unit vad is shorter than 0.2s
                continue

            organized_anno = [{
                "id": int(obj_id),
                "image_id": int(img_id),
                "category_id": int(category_id),
                "bbox": [
                    x1,
                    y1,
                    width,
                    height
                ],
                "area": float(width * height),
                "iscrowd": 0
            }]
            annotations.extend(organized_anno)

            det = [x1, y1, width, height, category_id]
            dets.append(det)
            obj_id += 1

        if len(dets) == 0:
            continue

        img.save(img_path)
        if mode == 1:
            img = cv2.imread(img_path)
            h_img, w_img, c = img.shape
            images.append({'id': img_id, 'file_name': each_file[:-4]+'_'+str(sub_cnt)+'.jpg', 'height':h_img, 'width':w_img})
        else:
            images.append({'id': img_id, 'file_name': each_file[:-4]+'_'+str(sub_cnt)+'.jpg'})

        # Write overlay image for debug
        ##img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ##img_overlay = annot_overlay_stft(img, dets)
        ##cv2.imwrite('tmp_figs2/' + each_file + str(sub_cnt) + '.jpg', img_overlay)
        sub_cnt += 1
        img_id += 1

categories = [
    {
        "id": 0,
        "name": "M"
    },
    {
        "id": 1,
        "name": "W"
    },
    {
        "id": 2,
        "name": "C"
    }
]


merged_label['images'] = images
merged_label['annotations'] = annotations
merged_label['categories'] = categories
# %%
with open(train_label_merge_out, 'w') as new_f:
    json.dump(merged_label, new_f, indent=4)
# %%
