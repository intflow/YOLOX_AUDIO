#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Intflow, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import os
import json
import cv2
import numpy as np
from scipy.io import wavfile
import scipy.io
import librosa
from PIL import Image

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES, INTFLOW_CLASSES, CROWDHUMAN_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis_bbox


WAV_EXT = [".wav"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX_AUDIO Demo!")
    parser.add_argument(
        "--demo", default="wav", help="demo type, eg. wav"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default="yolox_audio")
    parser.add_argument("-n", "--name", type=str, default="yolox_audio_x", help="model name")

    parser.add_argument(
        "--path", default="/data/AIGC_3rd_2021", help="path to images or video"
        #"--path", default="assets/00000.wav", help="path to images or video"
    )
    parser.add_argument(
        "--save_folder", default=None, help="path to images or video output"
    )
    parser.add_argument(
        "--multi_channel", default=None, help="set subsequent multi-channel files"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/yolox_audio/yolox_x.py",
        type=str,
        help="pls input your expriment description file",
    )
    #parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/yolox_audio/best_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument("-c", "--ckpt", default="/data/pretrained/yolox_x__AGC21_tr2.pth", type=str, help="ckpt for eval")
    #parser.add_argument("-m", "--model", default=None, type=str, help="model reference for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.8, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize_h", default=256, type=int, help="test img size(h)")
    parser.add_argument("--tsize_w", default=512, type=int, help="test img size(w)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--pruning",
        dest="pruning",
        default=False,
        action="store_true",
        help="Set pretrained model is whether pruned or not",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_wav_list(path):
    wav_names = []

    set_name = ['']

    file_name_list =[]
    for set_name_sub in set_name:
        path_sub = os.path.join(path, set_name_sub)
        for maindir, subdir, file_name_list in os.walk(path_sub):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in WAV_EXT:
                    wav_names.append(apath)
    return wav_names


class Predictor_audio(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        #if isinstance(img, str):
        #    img_info["file_name"] = os.path.basename(img)
        #    img = cv2.imread(img)
        #else:
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            ##logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        ###landmarks /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
##        rads = output[:,7]

        vis_res = vis_bbox(img, bboxes, scores, cls, cls_conf, self.cls_names)

        return vis_res

def sec2min(sec):
    m = int(sec / 60)
    s = int(sec % 60)
    out = str(m).zfill(2)+":"+str(s).zfill(2)

    return out

def wav_to_img(wav_path, multi_channel):
    
    try:
        if multi_channel == None:
            sr, wavs = wavfile.read(wav_path)
        else:
            wavs = []
            for mc in range(0,multi_channel):
                wav_path = wav_path.replace("ch"+str(mc),"ch"+str(mc+1))
                sr, wav_ch = wavfile.read(wav_path)
                wav_ch = wav_ch[:,0]
                wavs.append(wav_ch)
            wavs = np.array(wavs).T
    except:
        print('Can not load wav file')

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
        img = feat_cat_sub[...,::-1].copy()
        img_set.append(img)
        #img_path = os.path.join(img_folder_path, each_file)[:-4]+'_'+str(sub_cnt)+'.jpg'
        #img.save(img_path)
    return img_set, t_step, hop_length, sr

def wav_demo(predictor, vis_folder, path, current_time, save_result, multi_channel=None, save_folder=None):


    if os.path.isdir(path):
        files = get_wav_list(path)
    else:
        files = [path]
    files.sort()
    logger.info("load wav files")

    #define json outputs
    #json_file = open("track2/track2.json", "w")
    json_data = {}

    if multi_channel != None:
        files = files[::multi_channel]

    for _file in files:
        ## Load wav file then convert into image sets
        img_set, t_step, hop_length, sr = wav_to_img(_file, multi_channel)
        filename = _file.split('/')[-1]
        filename = filename.split('.')[0]

        img_idx = 0
        outputs_pixel_set = []
        for img in img_set:
            outputs, img_info = predictor.inference(img)
            outputs_pixel_set.append(outputs)
            ### Image save for visual debug
            ##result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            ##if save_result:
            ##    if save_folder == None:
            ##        save_folder = os.path.join(
            ##            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            ##        )
            ##    os.makedirs(save_folder, exist_ok=True)
            ##    image_name = filename+'_'+str(img_idx)+'.jpg'
            ##    save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            ##    #logger.info("Saving detection result in {}".format(save_file_name))
            ##    #cv2.imwrite(save_file_name, result_image)
            img_idx += 1
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        # Convert image-wise pixel vad into time values with json
        set_id = 0
        vad_set = []
        for outputs_pixel in outputs_pixel_set: 
            if outputs_pixel[0] != None:
                for vad_chunk in outputs_pixel[0]: # x0, y0, x1, y1, obj_score, cls_scre, cls_id
                    vad_pixel = [vad_chunk[0].item()+set_id, vad_chunk[2].item()+set_id, vad_chunk[5].item(), vad_chunk[6].item()] #we only use [x0, x1, cls_id]
                    vad_set.append(vad_pixel)
            set_id += t_step

        vad_set.sort(key = lambda x: x[0]) #Sort by time series

        #Merge adjacent VADs by distance and classes
        init = True
        del_list = []
        list_idx = 0
        for vad in vad_set:
            if init == True:
                init = False
                vad_1d = vad
            else:
                if vad[0] - vad_1d[1] <50.0:
                    if vad[2] > vad_1d[2]:
                        vad_prob = vad[2]
                        vad_cls = vad[3]
                    else:
                        vad_prob = vad_1d[2]
                        vad_cls = vad_1d[3]

                    vad_tmp = [vad_1d[0], vad[1], vad_prob, vad_cls]
                    vad_set[list_idx] = vad_tmp
                    del_list.append(list_idx-1)
                    vad_1d = vad_tmp
                else:
                    vad_1d = vad
            list_idx += 1

        #Delete duplicated list
        for del_id in sorted(del_list, reverse=True):
            del vad_set[del_id]

        #Post-processing
        time_unit = hop_length / sr #21.333333ms
        del_list = []
        del_id = 0
        for vad in vad_set:
            del_id += 1
            if vad[1] - vad[0] < 0.1/time_unit:
                del_list.append(del_id-1)
                continue
            if vad[0] <= 0.1/time_unit:
                del_list.append(del_id-1)
                continue

        for del_id in sorted(del_list, reverse=True):
            del vad_set[del_id]

        #Convert frame to time
        vad_set = np.asarray(vad_set)
        if len(vad_set) > 0:
            vad_set[:,0:2] *= time_unit


        #Write json
        speaker_lst = []
        on_offset_lst = []
        CLASS_NAME = ["M", "W", "C"]
        for vad in vad_set:
            speaker_lst.append(CLASS_NAME[int(vad[-1])])
            on_offset_lst.append([vad_set[0], vad_set[1]])

        json_data[_file+'.wav'] = {}
        json_data[_file+'.wav']["speaker"] = speaker_lst
        json_data[_file+'.wav']["on_offset"] = on_offset_lst
        #==== AIGC style ====
        ###if multi_channel is not None:
        ###    #Write json
        ###    if set_name not in json_data[0]["task2_answer"][0]:
        ###        json_data[0]["task2_answer"][0][set_name]=[]
        ###    
        ###    if len(json_data[0]["task2_answer"][0][set_name]) == int(drone_name[-1])-1:
        ###        json_data[0]["task2_answer"][0][set_name].append({})
        ###        json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name] = [{"M":["NONE"], "W":["NONE"], "C":["NONE"]}]
###
###
        ###    for vad_unit in vad_set:
        ###        vad_sub = []
        ###        vad_sub.append(sec2min(0.5*(vad_unit[0] + vad_unit[1])))
        ###        vad_sub.append(vad_unit[3])
###
        ###        if vad_sub[-1] == 0:
        ###            json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['M'].append(vad_sub[0])
        ###        elif vad_sub[-1] == 1:
        ###            json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['W'].append(vad_sub[0])
        ###        elif vad_sub[-1] == 2:
        ###            json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['C'].append(vad_sub[0])
###
        ###    if len(json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['M']) > 1:
        ###        del(json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['M'][0])
        ###    if len(json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['W']) > 1:
        ###        del(json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['W'][0])
        ###    if len(json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['C']) > 1:
        ###        del(json_data[0]["task2_answer"][0][set_name][int(drone_name[-1])-1][drone_name][0]['C'][0])

        #with open("track2/track2.json", "w", encoding='UTF-8') as json_file:
        #    json.dump(json_data, json_file, indent=2, ensure_ascii=False)

    logger.info("write json finished")

    return json_data
                

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    ###os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    ##if args.save_result:
    ##    vis_folder = os.path.join(file_name, "vis_res")
    ##    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    ##logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize_h is not None:
        exp.test_size = (args.tsize_h, args.tsize_w)


    if args.pruning:
        model = exp.get_model_pruning(args.model)
        ##logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    else:
        model = exp.get_model()
        ##logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        ##logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        ##logger.info("loaded checkpoint done.")

    if args.fuse:
        ##logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        ##logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    classes_list = INTFLOW_CLASSES
    predictor = Predictor_audio(model, exp, classes_list, trt_file, decoder, args.device, args.fp16, args.legacy)
    current_time = time.localtime()
    return wav_demo(predictor, vis_folder, args.path, current_time, args.save_result, args.multi_channel, args.save_folder)

def run_audio_infer(DATA_PATH):
    args = make_parser().parse_args()
    args.path  = DATA_PATH
    exp = get_exp(args.exp_file, args.name)

    return main(exp, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)