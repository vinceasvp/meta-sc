"""
-------------------------------File info-------------------------
% - File name: load_nsynth_data.py
% - Description: 根据不同的设置，读取nsynth数据集中的音频样本，并在线转换为时频特征
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-05-25
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: ee_w.xie@mail.scut.edu.cn
------------------------------------------------------------------
"""
import argparse
import os
import numpy as np
import random
import pickle
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import torchaudio
import pandas as pd
import json


# - 该函数是将全部数据的标签进行归类，使得同一类别的标签索引都归到一起，这样就可以知道对应的
#  - 数据集里总共有多少类别以及，每个类别内有多少个样本,
def build_label_index(label_unique_list):
    label2inds = defaultdict(list)
    num_labels = len(label_unique_list)
    for idxs, label_unique in enumerate(label_unique_list):  # - 对每个样本的标签逐一进行处理
        for label in label_unique:  # - 每个样本的标签有单个或者多个，也逐一进行处理
            # print(f'label_unique:{label_unique}')
            if label not in label2inds:  # - 对于每一种类别，都初始化一个存储空间
                label2inds[label] = []
            label2inds[label].append(idxs)  # - 把原始的类别编码所谓key,把样本所在的行号作为value
        # - 把同类样本的标识好都归到一起

    return label2inds


def load_meta(file):
    with open(file, 'rb') as fo:
        meta = pickle.load(fo)
    return meta


def wave_to_tfr(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    # - 直接在这里进行重采样
    # transform = torchaudio.transforms.Resample(sr, 16000)
    # waveform = transform(waveform)
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=128, dither=0.0,
                                              frame_shift=10)
    fbank = fbank.view(1, fbank.shape[0], fbank.shape[1])
    return fbank

    # n_fft = 1024
    # win_length = None
    # hop_length = 512
    # n_mels = 64
    #
    # mel_spectrogram = T.MelSpectrogram(
    #     sample_rate=sr,
    #     n_fft=n_fft,
    #     win_length=win_length,
    #     hop_length=hop_length,
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     norm='slaney',
    #     onesided=True,
    #     n_mels=n_mels,
    #     mel_scale="htk",
    # )
    #
    # melspec = mel_spectrogram(waveform)
    #
    # return melspec


class NsynthDatasets(Dataset):
    def __init__(self, _args, phase=None):
        self.phase = phase
        self.audio_dir = _args.audiopath
        self.meta_dir = os.path.join(_args.metapath, 'nsynth-' + str(_args.num_class) + '-fs-meta')
        with open(os.path.join(self.meta_dir, 'nsynth-' + str(_args.num_class) + '-fs_vocab.json')) as vocab_json_file:
            label_to_ix = json.load(vocab_json_file)

        if self.phase == 'train':
            meta_info = pd.read_csv(os.path.join(self.meta_dir, 'nsynth-' + str(_args.num_class) + '-fs_train.csv'))

        elif self.phase == 'val':
            meta_info = pd.read_csv(os.path.join(self.meta_dir, 'nsynth-' + str(_args.num_class) + '-fs_val.csv'))

        elif self.phase == 'test':
            meta_info = pd.read_csv(os.path.join(self.meta_dir, 'nsynth-' + str(_args.num_class) + '-fs_test.csv'))
        else:
            raise Exception('No such phase {0}, only support train, val and test'.format(phase))

        self.filenames = meta_info['filename']
        self.labels = meta_info['instrument']
        self.audio_source = meta_info['audio_source']
        label_code = []
        # 将标签字符串转换为数字编码
        for i in range(len(self.labels)):
            label_code.append(label_to_ix[self.labels[i]])

        self.label_codes = np.array(label_code)  # - 这一步可以把list中元素间的逗号去掉。即[0,1,2,...88] ->[0 1 2  88]
        # print(f'check labels: {self.label_codes}')
        self.sub_indexes = defaultdict(list)
        target_max = np.max(self.label_codes)  # - 由于标签都是数字编码，因此直接取最大的，这里需要改进

        for i in range(target_max + 1):
            self.sub_indexes[i] = np.where(self.label_codes == i)[0]  # - 将同一标签的行索引都归到一起

    def __getitem__(self, index):

        audio_feature = wave_to_tfr(os.path.join(self.audio_dir, self.audio_source[index], 'audio',
                                                 self.filenames[index] + '.wav'))
        label_out = self.label_codes[index]

        return audio_feature, label_out

    def __len__(self):
        return len(self.filenames)


def nsynth_dataset_for_fscil(args_):
    # - 将总的100类进行分组，0~54作为session0的类，随后每way个类作为一组，对应每个session,
    # - 从而得到 [ [0,1,..54],[55,56,57,58,59],...[95,96,97,98,99]]
    label_per_session = [list(np.array(range(args_.base_class)))] + \
                        [list(np.array(range(args_.way)) + args_.way * task_id + args_.base_class)
                         for task_id in range(args_.tasks)]

    dataset_train = NsynthDatasets(args_, phase='train')
    dataset_val = NsynthDatasets(args_, phase='val')
    dataset_test = NsynthDatasets(args_, phase='test')
    # 调用以上两个数据集，每次会返回一张声谱图（48, 1, 398, 128）, 和一个数字标签，e.g. 19

    train_datasets = []
    test_datasets = []

    all_datasets = {}

    # - 将训练集和测试集根据每个session所包含的类别进一步拆分成多个子数据集
    # - 最终对于每个session，都对应有一个训练集和一个测试集
    for session_id in range(args_.session):

        train_datasets.append(SubDatasetTrain(dataset_train, label_per_session, args_, session_id))
        test_datasets.append(SubDatasetTest(dataset_test, label_per_session, session_id))

    all_datasets['train'] = train_datasets
    all_datasets['val'] = dataset_val  # 验证集只有一个，且仅在session 0的训练时用到，因此不需要划分，直接返回即可
    all_datasets['test'] = test_datasets

    return all_datasets


class SubDatasetTrain(Dataset):
    def __init__(self, dataset, sublabels, args__, task_ids):
        self.ds = dataset
        self.indexes = []
        self.sub_indexes = defaultdict(list)
        if task_ids == 0:
            # 单独处理基类
            sublabel = sublabels[task_ids]  # - 把基类的标签取出
            # - dataset类有一个属性sub_indexes，它类似一个字典，根据类标签来把对应样本的索引保存起来
            # - 打印dataset_train.sub_indexes[99]可以看到：array([97, 146,301,...])
            #  - 这里的样本索引号是在整个数据集内全局唯一的
            # - 因此，可以按照标签来把对应的样本索引取出，从而实现对数据集的进一步划分
            for label in sublabel:
                self.indexes.extend(dataset.sub_indexes[int(label)])  # - 对于基类（session 0）取出全部的索引
                self.sub_indexes[label] = dataset.sub_indexes[int(label)]  # - 这个list在CategoriesSampler中被调用
        else:
            # - 这里的for循环在每次拆分新类的时候都会重复之前的，覆盖之前
            # 从第一个session开始都是增量session
            # for task in range(1, task_ids + 1):
            sublabel = sublabels[task_ids]
            # - 对于增量session 则每个session内每个类别随机采args.shot个样本
            for label in sublabel:

                shot_sample = random.sample(list(dataset.sub_indexes[int(label)]), args__.shot)

                self.indexes.extend(shot_sample)
                self.sub_indexes[label] = shot_sample

    def __getitem__(self, item):
        return self.ds[self.indexes[item]]  # - 类似常规的数据加载，但对应的索引已经被重定向

    def __len__(self):
        return len(self.indexes)


class SubDatasetTest(Dataset):
    def __init__(self, dataset, sublabels, task_ids):
        self.ds = dataset
        self.sub_indexes = []
        # - task是从0开始
        # - cifar100数据集有100个类，每个类里有500张训练图像，100张测试图像，因此这两个初始的数据子集都拥有100个类的数据
        # - 第一次调用本函数，是采样基类的全部测试样本,从第二次开始就逐步累积，即不仅采样当前session内包含类对应的样本
        # - 还包含了之前session的对包含类的全部样本
        for task in range(task_ids + 1):
            sublabel = sublabels[task]
            for label in sublabel:
                self.sub_indexes.extend(dataset.sub_indexes[int(label)])

    def __getitem__(self, item):
        return self.ds[self.sub_indexes[item]]

    def __len__(self):
        return len(self.sub_indexes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metapath', type=str, required=True, help='path to nsynth-xxx-fs-meta folder')
    parser.add_argument('--audiopath', type=str, required=True, help='path to The NSynth Dataset folder)')
    parser.add_argument('--num_class', type=int, default=100, help='Total number of classes in the dataset')

    # dataset setting(class-division, way, shot)
    parser.add_argument('--base_class', type=int, default=55, help='number of base class (default: 60)')
    parser.add_argument('--way', type=int, default=5, help='class number of per task (default: 5)')
    parser.add_argument('--shot', type=int, default=5, help='shot of per class (default: 5)')

    # hyper option
    parser.add_argument('--session', type=int, default=10, metavar='N',
                        help='num. of sessions, including one base session and n incremental sessions (default:10)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    args.tasks = args.session - 1  # 增量session的个数

    train_dataset = NsynthDatasets(args, phase='train')
    val_dataset = NsynthDatasets(args, phase='val')
    test_dataset = NsynthDatasets(args, phase='test')

    # print(f'sub_index:{train_dataset.sub_indexes.keys()}')
    # print(f'sub_index:{val_dataset.sub_indexes.keys()}')
    # print(f'sub_index:{test_dataset.sub_indexes.keys()}')

    # for i in range(len(train_dataset.sub_indexes.keys())):
    #     print(f'{i} - {len(train_dataset.sub_indexes[i])}')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True,
                                               num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=48 * 2, shuffle=False,
                                              num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=48, shuffle=True,
                                             num_workers=32, pin_memory=True)
    # loop all the batch
    num_epochs = 2

    # data_loader = train_loader
    # data_loader = val_loader
    data_loader = test_loader
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(data_loader):
            # - unpack data
            fea_batch, label_batch = batch_data
            # forward backward ,update, etc.
            if (batch_idx + 1) % 2 == 0:
                print(f'epoch {epoch + 1}/{num_epochs},'
                      f'features shape : {fea_batch.shape},'
                      f' label: {label_batch}\n'
                      )
    print('done.\n\n\n\n')
    # - 以上展示的按照常规方式读取数据集

    # 再FSCIL的任务中，需要将数据集进一步按类别划分，使得在每个session内，读取相应类别的数据，具体操作如下：
    # 1: 定义任务数据集函数
    datasets = nsynth_dataset_for_fscil(args)

    # 然后在 session i 过程中的训练子集、验证子集、测试子集 如下：
    i = 1
    trainset_i = datasets['train'][i]
    valset_0 = datasets['val'] # - 只有基类才有验证数据，因此验证集只在session 0出现
    testset_i = datasets['test'][i]
