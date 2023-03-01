import os
import json
from glob import glob
from pydub import AudioSegment
from tqdm import tqdm

data_dir = "/data/datasets/librispeech_fscil/train-clean-100"
spk_total_duration_json_path = "data/librispeech/spk_total_duration.json"
output_dir = "/data/datasets/librispeech_fscil/spk_single_audio"
wav_path_list = sorted(glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
num_select_spk = 100
if os.path.exists(spk_total_duration_json_path):
    with open(spk_total_duration_json_path, 'r') as json_file:
        spk_total_duration = json.load(json_file)
selected_spk_id = list(spk_total_duration.keys())[:num_select_spk]
selected_spk_td = list(spk_total_duration.values())[:num_select_spk]
for spk in tqdm(selected_spk_id, total=len(selected_spk_id)):
    spk_audio = AudioSegment.empty()
    for wav_path in wav_path_list:
        if spk == wav_path.split('/')[-3]:
            spk_audio += AudioSegment.from_wav(wav_path)
    spk_audio.export(f"/data/datasets/librispeech_fscil/spk_single_audio/{spk}.wav", format='wav')