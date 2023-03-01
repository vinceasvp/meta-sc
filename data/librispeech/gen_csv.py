import os
from select import select
import pandas as pd
from glob import glob
import librosa
import json
from tqdm import tqdm

data_dir = "/data/datasets/librispeech_fscil/train-clean-100"
spk_total_duration_json_path = "data/librispeech/spk_total_duration.json"
wav_path_list = sorted(glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
flac_path_list = sorted(glob(os.path.join(data_dir, "**/*.flac"), recursive=True))
num_select_spk = 100
# print(wav_path_list)
# print(flac_path_list)

for flac_path in flac_path_list:
    if os.path.exists(flac_path):
        os.remove(flac_path)
if os.path.exists(spk_total_duration_json_path):
    with open(spk_total_duration_json_path, 'r') as json_file:
        spk_total_duration = json.load(json_file)
else:
    spk_total_duration = {}
    spks = os.listdir(data_dir)
    for spk in spks:
        spk_total_duration[spk] = 0.0

    for wav_path in tqdm(wav_path_list):
        spk = wav_path.split('/')[-3]
        # if spk not in spk_total_duration.keys():
            # spk_total_duration[spk] = 0.0
        seg_audio, sr = librosa.load(wav_path, sr=None)
        spk_total_duration[spk] += len(seg_audio) / sr
    spk_total_duration = dict(sorted(spk_total_duration.items(), key = lambda x:x[1], reverse=True))
    print(spk_total_duration)
    with open(spk_total_duration_json_path, 'w') as json_file:
        json.dump(spk_total_duration, json_file, indent=2)

selected_spk_id = list(spk_total_duration.keys())[:num_select_spk]
selected_spk_td = list(spk_total_duration.values())[:num_select_spk]
print(selected_spk_id)