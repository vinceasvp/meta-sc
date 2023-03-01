import logging
import random
import torch
import os
import time
from copy import deepcopy
import numpy as np
import pprint as pprint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import pandas

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return round(self.v, 5)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.avg = round(self.sum / self.count, 5)
    
    def average(self):
        return self.avg


class LAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val):
        self.val = val
        self.count += 1
        if len(self.sum) == 0:
            assert (self.count == 1)
            self.sum = [v for v in val]
            self.avg = [round(v, 4) for v in val]
        else:
            assert (len(self.sum) == len(val))
            for i, v in enumerate(val):
                self.sum[i] += v
                self.avg[i] = round(self.sum[i] / self.count, 4)


class DAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}

    def update(self, values):
        assert (isinstance(values, dict))
        for key, val in values.items():
            if isinstance(val, (float, int)):
                if not (key in self.values):
                    self.values[key] = AverageMeter()
                self.values[key].update(val)
            elif isinstance(val, np.ndarray):
                val = float(val)
                if not (key in self.values):
                    self.values[key] = AverageMeter()
                self.values[key].update(val)
            elif isinstance(val, dict):
                if not (key in self.values):
                    self.values[key] = DAverageMeter()
                self.values[key].update(val)
            elif isinstance(val, list):
                if not (key in self.values):
                    self.values[key] = LAverageMeter()
                self.values[key].update(val)

    def average(self):
        average = {}
        for key, val in self.values.items():
            if isinstance(val, type(self)):
                average[key] = val.average()
            else:
                average[key] = val.avg
        return average


def acc_utils(da, num_base, num_session, way, session):
    acc_dict = {}
    acc_dict['all_acc'] = 0.0
    acc_dict['base_acc'] = 0.0
    acc_dict['novel_acc'] = 0.0
    acc_dict['former_acc'] = 0.0
    acc_dict['cur_acc'] = 0.0
    for i in range(num_session):
        acc_dict['sess{}_acc'.format(i)] = None
    if session == 0:
        avger = Averager()
        for i in range(num_base):
            if i in da:
                avger.add(da[i])
        acc_dict['all_acc'] = avger.item()
        acc_dict['former_acc'] = None
        acc_dict['cur_acc'] = avger.item()
        acc_dict['base_acc'] = avger.item()
        acc_dict['novel_acc'] = None
        acc_dict['sess0_acc'] = avger.item()
    else:
        for i in range(session + 1):
            if i == 0:
                sess_cls = range(num_base)
                acc_dict['sess{}_acc'.format(i)] =  get_aver(sess_cls, da)
            else:
                sess_cls = range(num_base + way * (i - 1), num_base + way * i)
                acc_dict['sess{}_acc'.format(i)] =  get_aver(sess_cls, da)

        all_cls = range(num_base + way * session)
        former_cls = range(num_base + way * (session - 1))
        cur_cls = range(num_base + way * (session - 1), num_base + way * session)
        base_cls = range(num_base)
        novel_cls = range(num_base, num_base + way * session)

        acc_dict['all_acc'] = get_aver(all_cls, da)
        acc_dict['former_acc'] = get_aver(former_cls, da)
        acc_dict['cur_acc'] = get_aver(cur_cls, da)
        acc_dict['base_acc'] = get_aver(base_cls, da)
        acc_dict['novel_acc'] = get_aver(novel_cls, da)
    return acc_dict

def cal_auxIndex(final_out_dict, alpha=0.5):
    aux_index = {}
    acc_aver = {}
    acc_aver['acc_cur_aver'] = 0.0
    acc_aver['acc_base_aver'] = 0.0
    acc_aver['acc_novel_aver'] = 0.0
    acc_aver['acc_both_aver'] = 0.0
    ar_over = ((final_out_dict['base_Acc'][0] - final_out_dict['base_Acc'][-1]) / final_out_dict['base_Acc'][0])
    msr_over = 1 - ar_over
    acc_aver['acc_base_aver'] = sum(final_out_dict['base_Acc']) / len(final_out_dict['base_Acc'])
    acc_aver['acc_both_aver'] = sum(final_out_dict['Both_ACC']) / len(final_out_dict['Both_ACC'])
    if len(final_out_dict['novel_Acc']) - 1 > 0:
        acc_aver['acc_novel_aver'] = sum(final_out_dict['novel_Acc'][1:]) / (len(final_out_dict['novel_Acc']) - 1)
        acc_aver['acc_cur_aver'] = sum(final_out_dict['cur_acc'][1:]) / (len(final_out_dict['cur_acc']) - 1)
    else:
        acc_aver['acc_novel_aver'] = None
        acc_aver['acc_cur_aver'] = None
    if acc_aver['acc_novel_aver'] is not None:
        cpi = alpha * msr_over + (1 - alpha) * acc_aver['acc_novel_aver']
    else:
        cpi = None
    acc_df = pandas.Series(acc_aver)
    return cpi, msr_over, acc_df, ar_over

def cal_auxIndex_from_numpy(acc_array, alpha=0.5):
    aux_index = {}
    acc_aver = {}
    acc_aver['acc_cur_aver'] = 0.0
    acc_aver['acc_base_aver'] = 0.0
    acc_aver['acc_novel_aver'] = 0.0
    acc_aver['acc_both_aver'] = 0.0
    ar_over = ((acc_array[1][0] - acc_array[1][-1]) / acc_array[1][0])
    msr_over = 1 - ar_over
    acc_aver['acc_base_aver'] = sum(acc_array[1]) / len(acc_array[1])
    acc_aver['acc_both_aver'] = sum(acc_array[3]) / len(acc_array[3])
    if len(acc_array[2]) - 1 > 0:
        acc_aver['acc_novel_aver'] = sum(acc_array[2][1:]) / (len(acc_array[2]) - 1)
        acc_aver['acc_cur_aver'] = sum(acc_array[0][1:]) / (len(acc_array[0]) - 1)
    else:
        acc_aver['acc_novel_aver'] = None
        acc_aver['acc_cur_aver'] = None
    if acc_aver['acc_novel_aver'] is not None:
        cpi = alpha * msr_over + (1 - alpha) * acc_aver['acc_novel_aver']
    else:
        cpi = None
    acc_df = pandas.Series(acc_aver)
    return cpi, msr_over, acc_df, ar_over

def get_aver(cls, da):
    avger = Averager()
    for i in cls:
        if i in da:
            avger.add(da[i])
    return avger.item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_per_cls_acc(logits, true_label):
    pred = torch.argmax(logits, dim=1)
    acc_dict = {}
    cls_sample_count = {}
    for cls in true_label.unique():
        indices = torch.where(true_label==cls, 1, 0)
        idx = torch.nonzero(indices)
        if torch.cuda.is_available():
            per_acc = (pred == true_label)[idx].type(torch.cuda.FloatTensor).mean().item()
        else:
            per_acc = (pred == true_label)[idx].type(torch.FloatTensor).mean().item()
        acc_dict[cls.cpu().data.item()] = per_acc
        cls_sample_count[f"number:{cls.cpu().data.item()}"] = len(idx)
        # cls_sample_count[f"percent:{cls.cpu().data.item()}"] = len(idx) / len(indices)
    return acc_dict, cls_sample_count
    
def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1) 
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)

def count_acc_taskIL(logits, label,args):
    basenum=args.base_class
    incrementnum=(args.num_classes-args.base_class)/args.way
    for i in range(len(label)):
        currentlabel=label[i]
        if currentlabel<basenum:
            logits[i,basenum:]=-1e9
        else:
            space=int((currentlabel-basenum)/args.way)
            low=basenum+space*args.way
            high=low+args.way
            logits[i,:low]=-1e9
            logits[i,high:]=-1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


def get_base_novel_ids(meta_labels, num_base, episode_way, session=0):
    batch_size = meta_labels.size(0)
    all_base_labels = np.arange(num_base+ session * episode_way)
    novel_ids = np.unique(meta_labels.cpu().numpy())
    base_ids = np.delete(all_base_labels, novel_ids)
    base_ids = torch.from_numpy(base_ids)
    novel_ids = torch.from_numpy(novel_ids)
    base_ids = base_ids.type(torch.cuda.LongTensor)
    novel_ids = novel_ids.type(torch.cuda.LongTensor)
    return base_ids, novel_ids

class Logger:
    def __init__(self, savedir):
        self.log = logging.getLogger(savedir)
        self.log.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s: %(message)s")

        self.fhander = logging.FileHandler(os.path.join(savedir, 'logs.log'), mode='w')
        self.fhander.setLevel(logging.DEBUG)
        self.fhander.setFormatter(self.formatter)
        self.log.addHandler(self.fhander)

        self.shander = logging.StreamHandler()
        self.shander.setLevel(logging.DEBUG)
        self.shander.setFormatter(self.formatter)
        self.log.addHandler(self.shander)

    def write_log(self, *args, **kwargs):
        self.log.info(args)
        # self.log.info(kwargs)


def get_torch_size(model, input_size):
    import torchinfo
    model_profile= torchinfo.summary(model, input_size=input_size)
    return model_profile.total_mult_adds, model_profile.total_params

def set_zeros_for_batch_audio(raw_batch_audio, start_frame, zero_length):
    tmp_batch_audio = deepcopy(raw_batch_audio)
    if start_frame >=0:
        tmp_batch_audio[:, start_frame:start_frame + zero_length] = 0.0
    return tmp_batch_audio

