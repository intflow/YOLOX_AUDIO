seed = 777
# SMT dataset utterance index start with 1, however our Lab. dataset utterance index start with 0
# UTTERANCE_BIAS = 1 # for SMT
UTTERANCE_BIAS = 1 # for our Lab(MA000, MK000, FA000, FK000)

from multiprocessing import Pool, TimeoutError
from tqdm import *
import glob
import numpy as np
import sys, os, time
import json
import numpy as np
import subprocess
import librosa
import scipy
import scipy.io.wavfile
from scipy.signal import fftconvolve
import argparse

class Generator():
    def __init__(self, source_dir, event_cat, name_path, data_dir, rir_dir, data_type = 'train', fs = 48000, num_channel = 7, length = 30, gain = 0.3, seed = 777, index_bias = 0):

        self.event_dir = os.path.join(source_dir, 'event')
        self.org_noise_dir  = os.path.join(source_dir, 'background')
        self.data_dir = data_dir
        self.rir_dir = rir_dir
        self.data_type = data_type
        self.event_cat = event_cat
        
        f_name = open(name_path, "r")
        names = f_name.readlines()
        self.names = list(map(lambda s: s.strip(), names))
        self.check_directory()
        self.config_session()
        self.mix_conifg()

        self.num_channel = num_channel
        self.fs = fs
        self.length = length
        self.gain = gain
        self.maxv = np.iinfo(np.int16).max
        self.index_bias = index_bias
        
        self.noises = self.load_noise(self.org_noise_dir)
        self.rirs   = self.load_rir(self.rir_dir)
        self.event_session_, self.event_wav_, self.event_type_ = self.config_session()
        self.seed = seed

    def mix_conifg(self):
        # 음원 간격
        self.sil_ = np.add(np.multiply(np.arange(19), 1), 1) # 1 ~ 20sec
        # 음원 개수 및 확률
        self.num_mix_ = np.arange(60) # mix 0 ~ 9
        self.num_mix_prob_ = self.num_mix_ / np.sum(self.num_mix_)

        # self.ov_prob = 0.15
        self.ov_prob = 0.3
        # self.ov_len_prob = np.array([.10, .20, .30, .40, .50, .60])
        self.ov_len_prob = np.array([.40, .50, .60, .70])
        
        self.num_nosie_prob_ = np.array([.05, .25, .25, .20, .15, .10])

    def load_event(self, event_dir):
        self.num_max_acoutics = 10
        self.min_ac_interval = 24000 # 0.5 sec
        noise_paths = glob.glob(os.path.join(event_dir, '*.wav'), recursive = False)
        noise_wavs = []
        for _ in range(len(noise_paths)): noise_wavs.append([])
        for n in range(len(noise_paths)):
            wav, _ = librosa.load(noise_paths[n], mono = False, sr = self.fs)
            if len(wav.shape) == 1:
                wav = np.tile(wav, (self.num_channel, 1))
            elif len(wav.shape) == 2:
                if wav.shape[0] != self.num_channel:
                    wav = np.tile(wav[0, :], (self.num_channel, 1))
            noise_wavs[n] = wav
        return noise_wavs    
    
    def load_noise(self, event_dir):
        self.noise_pertb_length = 240
        noise_paths = glob.glob(os.path.join(event_dir, '*.wav'), recursive = False)
        noise_wavs = []
        for _ in range(len(noise_paths)): noise_wavs.append([])
        for n in range(len(noise_paths)):
            wav, _ = librosa.load(noise_paths[n], mono = False, sr = self.fs)
            if len(wav.shape) == 1:
                wav = np.tile(wav, (self.num_channel, 1))
            elif len(wav.shape) == 2:
                if wav.shape[0] != self.num_channel:
                    wav = np.tile(wav[0, :], (self.num_channel, 1))
            noise_wavs[n] = wav
        return noise_wavs
        
    def load_rir(self, rir_dir):    
        if rir_dir is not None:
            rir_paths = glob.glob(os.path.join(rir_dir, '*.wav'), recursive = False)
            rir_wavs = []
            for _ in range(len(rir_paths)): rir_wavs.append([])
            for n in range(len(rir_paths)):
                wav, _ = librosa.load(rir_paths[n], mono = False, sr = self.fs)
                if (len(wav.shape) == 1) or (wav.shape[0] != self.num_channel):
                    raise TypeError('check rir dataset')
                rir_wavs[n] = wav
            return rir_wavs
        else:
            return None

    def config_session(self):
        if self.data_type == 'train':
            animal_cry_sessions   = ['pigfarm_cry_01',]
            animal_cry_pathes = []
            for idx in range(len(animal_cry_sessions)):
                paths = glob.glob(os.path.join(os.path.join(self.event_dir, animal_cry_sessions[idx]), '*.wav'))
                animal_cry_pathes.append(paths)
            animal_cry_pathes = np.asarray(animal_cry_pathes)
            
            farm_noise_sessions = ['pigfarm_hitnoise_01',]
            farm_noise_pathes = []
            for idx in range(len(farm_noise_sessions)):
                paths = glob.glob(os.path.join(os.path.join(self.event_dir, farm_noise_sessions[idx]), '*.wav'))
                farm_noise_pathes.append(paths)
            farm_noise_pathes = np.asarray(farm_noise_pathes)
            
            ##--- You can add other event types ----
            ####child_sessions = ['AC100', 'AC102', 'AC103', 'AC104', 'AC105', 'AC106', 'AC107', 'AC108', 'AC109']
            ####
            ####num_child_wav = []
            ####for idx in range(len(child_sessions)):
            ####    paths = glob.glob(os.path.join(os.path.join(self.event_dir, child_sessions[idx]), '*.wav'))
            ####    num_child_wav.append(len(paths))
            ####num_child_wav = np.asarray(num_child_wav)

        
        session_list = [animal_cry_sessions, farm_noise_sessions]
        wav_list     = [animal_cry_pathes, farm_noise_pathes]
        type_list   = ['cry', 'noise']
        
        return session_list, wav_list, type_list
        
    def process(self, file_index):
        np.random.seed(file_index + self.index_bias)
        while True:
            num_src = np.random.choice(len(self.num_mix_), 1, p = self.num_mix_prob_)[0]

            ## Make unit length background signals
            sig_tmp = []
            sig_tot = 0
            unit_len = int(round(self.length * self.fs))
            while True:
                rand_noise = self.noises[np.random.randint(len(self.noises))]
                rand_idx = np.random.randint(len(rand_noise.T))
                sig_tot += len(rand_noise[0,rand_idx:])
                sig_tmp.append(rand_noise[:,rand_idx:])
                if sig_tot >= unit_len:
                    break 
            sig_tmp = np.concatenate(sig_tmp, axis=1)
            sig_tmp = sig_tmp[:,:unit_len]
            sig_tmp = np.multiply(sig_tmp, 0.6)
            
            ## Mix Event sounds into background signal
            event_class_list, event_session_list, on_offset_list = [], [], []
            silence_duration = int((self.sil_[np.random.randint(len(self.sil_))] + np.random.rand()) * self.fs)
            clean_sig = np.zeros((self.num_channel, silence_duration))
            for idx in range(num_src):
                event_idx = np.random.choice(len(self.event_type_), 1)[0]
                event_class = self.event_type_[event_idx]
                event_class_list.append(event_class)
                
                session_idx = np.random.choice(len(self.event_session_[event_idx]), 1)[0]

                session = self.event_session_[event_idx][session_idx]
                event_session_list.append(session)
                session_wav_path = '%s' %(np.random.choice(self.event_wav_[event_idx][session_idx], 1)[0])
                
                session_wav, _  = librosa.load(session_wav_path, mono = True, sr = self.fs)
                onset = 0
                offset = len(session_wav[:]) / self.fs
                seg_session_wav = session_wav[int(self.fs * onset):int(self.fs * offset)]
                
                #Write global on/offset
                data_onset  = float(clean_sig.shape[1]) / self.fs
                data_offset = float(clean_sig.shape[1]) / self.fs + offset - onset
                if data_onset >= self.length:
                    break
                on_offset_list.append([round(data_onset, 4), round(data_offset, 4), int(event_idx)])

                if self.rirs is not None:
                    rir_wav = self.rirs[np.random.randint(len(self.rirs))]
                
                    seg_sig = []
                    for ch_idx in range(self.num_channel):
                        seg_sig = np.append(seg_sig, fftconvolve(seg_session_wav, rir_wav[ch_idx, :]))
                else:
                    seg_sig = seg_session_wav
                
                seg_sig = np.reshape(seg_sig, (self.num_channel, -1))
                seg_sig = np.clip(seg_sig, -0.9, 0.9)
                seg_sig = np.multiply(seg_sig, self.gain)

                if self.ov_prob > np.random.rand():
                    ##while True:
                    ##    event_idx_temp = np.random.choice(len(self.event_type_), 1)[0]
                    ##    event_class_temp = self.event_type_[event_idx_temp]
                    ##    if event_class_temp != event_class:
                    ##        break
                    event_idx_temp = np.random.choice(len(self.event_type_), 1)[0]
                    event_class_temp = self.event_type_[event_idx_temp]
                    event_class_list.append(event_class_temp)                    
                    session_idx = np.random.choice(len(self.event_session_[event_idx]), 1)[0]
                    session = self.event_session_[event_idx][session_idx]
                    event_session_list.append(session)
                    session_wav_path = '%s' %(np.random.choice(self.event_wav_[event_idx][session_idx], 1)[0])
                    
                    session_wav, _  = librosa.load(session_wav_path, mono = True, sr = self.fs)
                    onset = 0
                    offset = len(session_wav[:]) / self.fs
                    seg_session_wav = session_wav[int(self.fs * onset):int(self.fs * offset)]

                    if self.rirs is not None:
                        rir_wav = self.rirs[np.random.randint(len(self.rirs))]
                    
                    seg_sig_temp = []
                    for ch_idx in range(self.num_channel):
                        if self.rirs is not None:
                            seg_sig_temp = np.append(seg_sig_temp, fftconvolve(seg_session_wav, rir_wav[ch_idx, :]))
                        else:
                            seg_sig_temp = np.append(seg_sig_temp, seg_session_wav)

                    seg_sig_temp = np.reshape(seg_sig_temp, (self.num_channel, -1))
                    seg_sig_temp = np.clip(seg_sig_temp, -0.9, 0.9)
                    seg_sig_temp = np.multiply(seg_sig_temp, self.gain)
                    
                    seg_sig_min_len = np.min([seg_sig.shape[1], seg_sig_temp.shape[1]])
                    ov_len = int(np.random.choice(self.ov_len_prob) * seg_sig_min_len)
                    ov_seg_sig_len = seg_sig.shape[1] + seg_sig_temp.shape[1] - ov_len

                    ov_seg_sig = np.zeros((self.num_channel, ov_seg_sig_len))
                    ov_seg_sig[:, :seg_sig.shape[1]] = np.add(ov_seg_sig[:, :seg_sig.shape[1]], seg_sig)

                    ov_seg_sig[:, -seg_sig_temp.shape[1]:] = np.add(ov_seg_sig[:, -seg_sig_temp.shape[1]:], seg_sig_temp)
                    
                    data_onset  = (float(clean_sig.shape[1] + seg_sig.shape[1] - ov_len)) / self.fs
                    data_offset = (float(clean_sig.shape[1] + seg_sig.shape[1] - ov_len)) / self.fs + offset - onset
                    if data_onset >= self.length:
                        break
                    on_offset_list.append([round(data_onset,4), round(data_offset,4), int(event_idx_temp)])
                    
                    seg_sig = ov_seg_sig
                
                silence_duration = int((self.sil_[np.random.randint(len(self.sil_))] + np.random.rand()) * self.fs)
            
                silence = np.zeros((self.num_channel, silence_duration))
                seg_sig = np.append(seg_sig, silence, axis = 1)
                clean_sig = np.append(clean_sig, seg_sig, axis = 1)
            
            # amp_pertb = (np.random.rand() * (1.4125 - 0.7079)) + 0.7079
            amp_pertb = (np.random.rand() + 0.5) * 2.5
            clean_sig = np.multiply(clean_sig, amp_pertb)
            len_clean_sig = clean_sig.shape[1]
            len_out_sig = sig_tmp.shape[1]
            #len_noise = sig_tmp.shape[1]

            # noise_pos = np.random.randint(len_noise - (len_clean_sig + 1))
            # noise_sig = np.zeros(clean_sig.shape)
            # noise_pos = np.random.randint(len_noise - (len_clean_sig + 1 + self.noise_pertb_length))
            # for c_idx in range(self.num_channel):
            #     noise_pos_ = noise_pos + np.random.randint(self.noise_pertb_length)
            #     noise_sig[c_idx, :] = noise[c_idx, noise_pos_:noise_pos_ + len_clean_sig]
            
            ####num_acoutics = np.random.randint(self.num_max_acoutics)
            ####num_acoutics = np.min([num_acoutics, int(len_noise / self.min_ac_interval) - 2])

            ####if num_acoutics > 0:
            ####    ac_range = np.arange(np.random.randint(self.min_ac_interval), len_clean_sig - (self.fs * 2), self.min_ac_interval)
            ####    num_acoutics = np.min([num_acoutics, len(ac_range)])
            ####    ac_pos = np.random.choice(ac_range, num_acoutics, replace = False)
            ####    for a_idx in ac_pos:
            ####        acoustic_noise = self.acoutics[np.random.randint(len(self.acoutics))]
            ####        noise_sig[:, a_idx:a_idx + acoustic_noise.shape[1]] = np.add(noise_sig[:, a_idx:a_idx + acoustic_noise.shape[1]], acoustic_noise)
            
            if len_clean_sig < len_out_sig:
                sig_tmp[:,:len_clean_sig] += clean_sig
            else:
                sig_tmp += clean_sig[:,:len_out_sig]
            if file_index < len(self.names):
                wav_name = '%s.wav' % (self.names[file_index])
            else:
                wav_name = '%05d.wav' %(file_index + self.index_bias)
                print("----File name is not defined for %d , set name text properly!!!----"% file_index)
            if len_clean_sig <= self.fs * self.length:
                scipy.io.wavfile.write(os.path.join(self.data_dir, wav_name), self.fs, (self.maxv * sig_tmp.T).astype(np.int16))
                # scipy.io.wavfile.write(os.path.join(self.event_dir, wav_name), self.fs, (self.maxv * noise_sig.T).astype(np.int16))
                break
        
        return wav_name, event_class_list, event_session_list, on_offset_list
        
    def seed_update(self):
        np.random.seed(self.seed)

    def check_directory(self):
        if os.path.isdir(self.data_dir) == False: os.makedirs(self.data_dir, exist_ok = True)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir',   '-i', type = str, required = False, default = '/NIA75_2022/pig/raw_audio/pigfarmA/SL_audio_pig_2022-10-06/split', help = 'Input data')
    ###parser.add_argument('--source-dir',   '-i', type = str, required = False, default = 'test_data/split_test', help = 'Input data')
    parser.add_argument('--event-cat',   '-e', type = str, required = False, default = ['cry', 'noncry'], help = 'Input data')
    parser.add_argument('--name-path',  '-m', type = str, required = False, default = '/NIA75_2022/pig/raw_audio/pigfarmA/SL_audio_pig_2022-10-06/split/name_pigcry60.txt', help = 'Output data')
    ###parser.add_argument('--name-path',  '-m', type = str, required = False, default = 'test_data/name_pigcry5.txt', help = 'Output data')
    parser.add_argument('--data-dir',  '-o', type = str, required = False, default = '/NIA75_2022/pig/raw_audio/pigfarmA/SL_audio_pig_2022-10-06/split/dataset-pigcry60', help = 'Output data')
    ###parser.add_argument('--data-dir',  '-o', type = str, required = False, default = 'test_data/dataset-pigcry5', help = 'Output data')
    parser.add_argument('--rir-dir',  '-r', type = str, required = False, default = None, help = 'RIR data')
    parser.add_argument('--bias',    '-b', type = int, required = False, default = 0, help = 'Index bias')
    parser.add_argument('--datatype','-d', type = str, required = False, default = 'train', help = 'Data type')
    parser.add_argument('--length',     '-l', type = int, required = False, default = 300, help = 'file length(s)')
    parser.add_argument('--fs',     '-f', type = int, required = False, default = 48000, help = 'Sampling rate(hz)')
    parser.add_argument('--num_channel',     '-c', type = int, required = False, default = 1, help = 'Number of audio channel')
    parser.add_argument('--num',     '-n', type = int, required = False, default = 5, help = 'The number of the dataset')
    args = parser.parse_args()
    
    source_dir = args.source_dir
    event_cat = args.event_cat
    name_path = args.name_path
    data_dir = args.data_dir
    rir_dir = args.rir_dir
    index_bias = args.bias
    datatype = args.datatype
    length = args.length
    fs = args.fs
    num_channel = args.num_channel
    num_dataset = args.num
    data_dir = data_dir+"_seed"+str(index_bias)
    
    gt_data = {}
    # eval_data['track2_results'] = []

    num_workers = 8
    
    generator = Generator(source_dir, event_cat, name_path, data_dir, rir_dir, data_type = datatype, length = length, fs = fs, num_channel = num_channel, index_bias = index_bias)
    
    with Pool(processes = num_workers) as pool:
        with tqdm(total = num_dataset) as pbar:
            for meta_data in tqdm(pool.imap_unordered(generator.process, range(num_dataset))):
                wav_name, event_class, session_class, on_offset = meta_data
                dev_data = {}
                dev_data['AUDIO'] = {}
                dev_data['AUDIO']['AUDIO_FILE_NAME'] = wav_name
                dev_data['AUDIO']['CRY_COUNT'] = event_class.count('cry')
                dev_data['AUDIO']['CRY_TIMESTAMP'] = [item for sublist in on_offset for item in sublist]

                with open(os.path.join(data_dir, wav_name.split('.wav')[0]+'.json'.format(index_bias)), 'w') as f:  
                    json.dump(dev_data, f, ensure_ascii = False, indent = '    ', sort_keys = False)

                pbar.update()
    
    

    '''
    with open(os.path.join(gt_dir, 'gist_tr2_gt.json'), 'w') as f:  
         json.dump(gt_data, f, ensure_ascii = False, indent = '    ', sort_keys = False)

    with open(os.path.join(gt_dir, 'gist_tr2_gt_for_devel.json'), 'w') as f:  
         json.dump(dev_gt_data, f, ensure_ascii = False, indent = '    ', sort_keys = False)

    with open(os.path.join(gt_dir, 'tr2_res_sample.json'), 'w') as f:  
         json.dump(eval_data, f, ensure_ascii = False, indent = '    ', sort_keys = False)
         
    print()
    '''