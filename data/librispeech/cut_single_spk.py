import os
from glob import glob
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
import json

single_spk_dir = "/data/datasets/librispeech_fscil/spk_single_audio"
spk_segment_dir = "/data/datasets/librispeech_fscil/spk_segments"
train_csv_path = "/data/caowc/FSCIL/data/librispeech/librispeech_fscil_train.csv"
val_csv_path = "/data/caowc/FSCIL/data/librispeech/librispeech_fscil_val.csv"
test_csv_path = "/data/caowc/FSCIL/data/librispeech/librispeech_fscil_test.csv"
spk_mapping_path = "/data/caowc/FSCIL/data/librispeech/spk_mapping.json"
seg_duration = 2000 # ms
num_train_seg = 500
num_val_seg = 150
num_test_seg = 100

single_spk_path_list = sorted(glob(os.path.join(single_spk_dir, "*.wav")))
spk_seg_path_list = sorted(glob(os.path.join(spk_segment_dir, "*.wav")))

# empty the seg dir
if len(spk_seg_path_list) > 0:
    for spk_seg_path in spk_seg_path_list:
        os.remove(spk_seg_path)

for sing_spk_path in tqdm(single_spk_path_list):
    spk = os.path.basename(sing_spk_path).split('.')[0]
    spk_audio = AudioSegment.from_wav(sing_spk_path)
    num_full_segs = len(spk_audio) // seg_duration
    for i in range(num_full_segs):
        tmp_audio = spk_audio[i * seg_duration: (i + 1) * seg_duration]
        tmp_audio.export(os.path.join(spk_segment_dir, f"{spk}_{i}.wav"), format='wav')


spks_list = [os.path.basename(spk_path).split('.')[0] for spk_path in single_spk_path_list]
spk_mapping = {}
for i, spk in enumerate(spks_list):
    spk_mapping[spk] = i
with open(spk_mapping_path, 'w') as json_flie:
    json.dump(spk_mapping, json_flie, indent=2)

spk_seg_path_list = sorted(glob(os.path.join(spk_segment_dir, "*.wav")))
train_fn_list = []
train_spk_list = []
train_label_list = []

val_fn_list = []
val_spk_list = []
val_label_list = []

test_fn_list = []
test_spk_list = []
test_label_list = []

for spk in tqdm(spks_list):
    this_spk_seg_path_list = sorted(glob(os.path.join(spk_segment_dir, f"{spk}*.wav")))
    this_spk_seg_fn_list = [os.path.basename(seg_path) for seg_path in this_spk_seg_path_list]
    train_fn_list.extend(this_spk_seg_fn_list[:num_train_seg])
    val_fn_list.extend(this_spk_seg_fn_list[num_train_seg: num_train_seg + num_val_seg])
    test_fn_list.extend(this_spk_seg_fn_list[num_train_seg + num_val_seg: num_train_seg + num_val_seg + num_test_seg])

    train_spk_list.extend([spk] * num_train_seg)
    val_spk_list.extend([spk] * num_val_seg)
    test_spk_list.extend([spk] * num_test_seg)

    train_label_list.extend([spk_mapping[spk]] * num_train_seg)
    val_label_list.extend([spk_mapping[spk]] * num_val_seg)
    test_label_list.extend([spk_mapping[spk]] * num_test_seg)

train_dict = {"filename": train_fn_list, "speaker_id": train_spk_list, "label": train_label_list}
val_dict = {"filename": val_fn_list, "speaker_id": val_spk_list, "label": val_label_list}
test_dict = {"filename": test_fn_list, "speaker_id": test_spk_list, "label": test_label_list}

train_csv = pd.DataFrame(train_dict)
val_csv = pd.DataFrame(val_dict)
test_csv = pd.DataFrame(test_dict)

train_csv.to_csv(train_csv_path, sep=',', index=False)
val_csv.to_csv(val_csv_path, sep=',', index=False)
test_csv.to_csv(test_csv_path, sep=',', index=False)