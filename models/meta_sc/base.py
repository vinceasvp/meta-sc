import abc
import torch
import os.path as osp
import os
import pandas
from dataloader.dataloader import *

from utils.utils import (
    ensure_path,
    Averager, Timer, count_acc, cal_auxIndex
)


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = self.set_up_datasets()
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] #* args.sessions
        self.sess_acc_dict = {}
        self.result_list = [args]


    def set_up_datasets(self):
        elif self.args.dataset == 'nsynth-100':
            import dataloader.nsynth.nsynth as Dataset
        elif self.args.dataset == 'nsynth-200':
            import dataloader.nsynth.nsynth as Dataset
        elif self.args.dataset == 'nsynth-300':
            import dataloader.nsynth.nsynth as Dataset
        elif self.args.dataset == 'nsynth-400':
            import dataloader.nsynth.nsynth as Dataset
        elif self.args.dataset == 'librispeech':
            import dataloader.librispeech.librispeech as Dataset
            import dataloader.s2s.s2s as Dataset
        self.args.Dataset=Dataset

    def pretty_output(self, save=True, print_output=True):
        final_out_dict = {}
        final_out_dict['cur_acc'] = []
        final_out_dict['base_Acc'] = []
        final_out_dict['novel_Acc'] = []
        final_out_dict['Both_ACC'] = []
        for k, v in self.sess_acc_dict.items():
            final_out_dict['cur_acc'].append(v['cur_acc'])
            final_out_dict['base_Acc'].append(v['base_acc'])
            if 'novel_acc' in v:
                final_out_dict['novel_Acc'].append(v['novel_acc'])
            else:
                final_out_dict['novel_Acc'].append(None)
            final_out_dict['Both_ACC'].append(v['all_acc'])
        cpi, msr_overall, acc_aver_df, ar_over = cal_auxIndex(final_out_dict)
        pd = final_out_dict['Both_ACC'][0] - final_out_dict['Both_ACC'][-1]
        indexes = {'PD':pd, 'CPI':cpi, 'AR':ar_over, 'MSR':msr_overall}
        indexes_df = pandas.DataFrame.from_dict(indexes, orient='index')
        final_df = pandas.DataFrame(final_out_dict)
        final_df = final_df.T
        # pretty output
        pandas.set_option('display.max_rows', None)
        pandas.set_option('display.max_columns', None)
        pandas.set_option('display.width', None)
        pandas.set_option('display.max_colwidth', None)
        if save:
            excel_fn = os.path.join(self.args.save_path, "output.xlsx")
            print("save output at ", excel_fn)
            writer = pandas.ExcelWriter(excel_fn)
            final_df.to_excel(writer, sheet_name='final_df')
            acc_aver_df.to_excel(writer, sheet_name='final_df', startrow=7)
            indexes_df.to_excel(writer, sheet_name='final_df', startrow=13)
            indexes_df.T.to_excel(writer, sheet_name='final_df', startrow=20)
            writer.save()

        output = f"\nreslut on {self.args.dataset}, method {self.args.project}\
                    \n{self.args.save_path}\
                    \n****************************************Pretty Output********************************************\
                    \n{final_df}\
                    \n===> Comprehensive Performance Index(CPI) v2: {cpi}\n===> PD: {pd}\
                    \n===> Memory Strock Ratio(MSR) Overall: {msr_overall}\n===> Amnesia Rate(AR): {ar_over}\
                    \n===> Acc Average: \n{acc_aver_df}\
                    \n***********************************************************************************************"
        if print_output:
            print(output)
        return output, acc_aver_df, final_df

    @abc.abstractmethod
    def train(self):
        pass