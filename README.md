# meta-sc



## Datasets

To study the Few-shot Class-incremental Audio Classification (FCAC) problem, LS-100 dataset and NSynth-100 dataset are constructed by choosing samples from audio corpora of the [Librispeech](https://www.openslr.org/12/) dataset, the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset respectively. Wei Xie, one of our team members, constructed the NSynth-100 dataset

The detailed information of the LS-100 dataset and NSynth-100 dataset are given below.

### Statistics on the LS-100 dataset

|                                                                 | LS-100                                        | NSynth-100                                    |
| --------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| Type of audio                                                   | Speech                                        | Musical instrumnets                           |
| Num. of classes                                                 | 100 (60 of base classes, 40 of novel classes) | 100 (55 of base classes, 45 of novel classes) |
| Num. of training / validation / testing samples per base class  | 500 / 150 / 100                               | 200 / 100 / 100                               |
| Num. of training / validation / testing samples per novel class | 500 / 150 / 100                               | 100 / none / 100                              |
| Duration of the sample                                          | All in 2 seconds                              | All in 4 seconds                              |

##### Preparation of the NSynth-100 dataset

The NSynth dataset is an audio dataset containing 306,043 musical notes, each with a unique pitch, timbre, and envelope. 
Those musical notes are belonging to 1,006 musical instruments.

Based on the statistical results, we obtain the NSynth-100 dataset by the following steps:

1. Download [Train set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz), [Valid set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz), and [test set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz) of the NSynth dataset to your local machine and unzip them.

2. Download the meta files for FCAC from [here](./data/nsynth) to your local machine and unzip them.

3. You will get s structure of the directory as follows:

   ```
   Your dataset root(Nsynth)
   ├── nsynth-100-fs-meta
   ├── nsynth-200-fs-meta
   ├── nsynth-300-fs-meta
   ├── nsynth-400-fs-meta
   ├── nsynth-test
   │   └── audio
   ├── nsynth-train
   │   └── audio
   └── nsynth-valid
       └── audio
   ```

   

### Preparation of the LS-100 dataset

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned. We find that the subset `train-clean-100` of Librispeech is enough for our study, so we constructed the LS-100 dataset using partial samples from the Librispeech as the source materials. To be specific, we first concatenate all the speakers' speech clips into a long speech, and then select the 100 speakers with the longest duration to cut their voices into two second speech. You can download the Librispeech from [here](https://www.openslr.org/12/).

1. Download [dataset](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and extract the files.

2. Transfer the format of audio files. Move the script `normalize-resample.sh` to the root dirctory of extracted folder, and run the command `bash normalize-resample.sh`.

3. Construct LS-100 dataset.
   
   ```
   python data/LS100/construct_LS100.py --data_dir DATA_DIR --duration_json data/librispeech/spk_total_duration.json --single_spk_dir SINGLE_SPK_DIR --num_select_spk 100 --spk_segment_dir SPK_SEGMENT_DIR --csv_path CSV_PATH --spk_mapping_path SPK_MAPPING_PATH
   ```

## Code

- run experiment on LS-100

    ```bash
    python train.py -project meta_sc -dataroot DATAROOT -dataset librispeech -lamda_proto 0.6 -config ./configs/meta_sc_LS-100_stochastic_classifier.yml -gpu 1
    ```

- run experiment on NS-100

  ```bash
  python train.py -project meta_sc -dataroot DATAROOT -dataset nsynth-100 -lamda_proto 0.6 -config ./configs/meta_sc_NS-100_stochastic_classifier.yml -gpu 1
  ```

  

## Acknowledgment

Our project references the codes in the following repos.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)

## Citation

Please cite our paper (shown below) if you find the codes and datasets are useful for your research.

[1] Yanxiong Li, Wenchang Cao, Jialong Li, Wei Xie, and Qianhua He, "Few-shot class-incremental audio classification using stochastic classifier," in Proc. of INTERSPEECH, Dublin, Ireland, 20-24 Aug., 2023.

